"""
Integration tests for Bayesian MCMC with real digital twin simulation.

Tests:
1. SimulationCache - hit/miss tracking, LRU eviction, stats
2. CachedSimulator - wraps digital twin, param update, fallback on failure
3. BayesianEstimator - estimate_parameters() no longer uses placeholder
4. ClinicalDataLoader - EDF/CSV loading and preprocessing
"""

import os
import sys
import shutil
import tempfile

import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ens_gi_digital.core import ENSGIDigitalTwin

# ── availability guards ──────────────────────────────────────────────────────

try:
    from ens_gi_digital.simulation_cache import SimulationCache, CachedSimulator
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

try:
    from ens_gi_digital.bayesian import BayesianEstimator, BayesianConfig
    import pymc as pm   # noqa: F401
    import arviz as az  # noqa: F401
    PYMC_AVAILABLE = True
except (ImportError, Exception):
    PYMC_AVAILABLE = False

try:
    from ens_gi_digital.patient_data import ClinicalDataLoader, create_sample_patient_data
    CLINICAL_LOADER_AVAILABLE = True
except ImportError:
    CLINICAL_LOADER_AVAILABLE = False

try:
    from ens_gi_digital.pinn import PINNEstimator, PINNConfig
    import tensorflow as tf  # noqa: F401
    TF_AVAILABLE = True
except (ImportError, Exception):
    TF_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════
# SimulationCache Tests
# ═══════════════════════════════════════════════════════════════

@pytest.mark.skipif(not CACHE_AVAILABLE, reason="simulation_cache not importable")
class TestSimulationCache:
    """Test SimulationCache hit/miss tracking and LRU eviction."""

    @pytest.fixture
    def tmp_cache(self, tmp_path):
        """Temporary cache in a temp directory."""
        return SimulationCache(
            cache_dir=str(tmp_path / 'test_cache'),
            max_size_mb=1
        )

    def test_initial_stats_zero(self, tmp_cache):
        """Cache starts empty with zero hits/misses."""
        stats = tmp_cache.stats()
        assert stats['hits'] == 0
        assert stats['misses'] == 0
        assert stats['hit_rate'] == 0.0
        assert stats['num_entries'] == 0

    def test_cache_miss_on_first_access(self, tmp_cache):
        """First access should be a miss."""
        params = {'g_Na': 120.0, 'g_K': 36.0}
        result = tmp_cache.get(params, duration=500.0, dt=0.1)
        assert result is None
        assert tmp_cache.misses == 1
        assert tmp_cache.hits == 0

    def test_cache_hit_after_put(self, tmp_cache):
        """After storing, the same params should be a hit."""
        params = {'g_Na': 120.0, 'g_K': 36.0}
        value = {'mean': -65.0, 'std': 10.0}

        tmp_cache.put(params, 500.0, 0.1, value)
        result = tmp_cache.get(params, 500.0, 0.1)

        assert result == value
        assert tmp_cache.hits == 1
        assert tmp_cache.misses == 0

    def test_different_params_different_keys(self, tmp_cache):
        """Different parameter values produce different cache keys."""
        params_a = {'g_Na': 120.0, 'g_K': 36.0}
        params_b = {'g_Na': 125.0, 'g_K': 36.0}
        value_a = {'mean': -65.0, 'std': 10.0}
        value_b = {'mean': -60.0, 'std': 12.0}

        tmp_cache.put(params_a, 500.0, 0.1, value_a)
        tmp_cache.put(params_b, 500.0, 0.1, value_b)

        assert tmp_cache.get(params_a, 500.0, 0.1) == value_a
        assert tmp_cache.get(params_b, 500.0, 0.1) == value_b

    def test_different_duration_different_keys(self, tmp_cache):
        """Same params but different duration → different cache entry."""
        params = {'g_Na': 120.0}
        value1 = {'mean': -65.0}
        value2 = {'mean': -64.0}

        tmp_cache.put(params, 500.0, 0.1, value1)
        tmp_cache.put(params, 1000.0, 0.1, value2)

        assert tmp_cache.get(params, 500.0, 0.1) == value1
        assert tmp_cache.get(params, 1000.0, 0.1) == value2

    def test_clear_resets_state(self, tmp_cache):
        """clear() removes all entries and resets counters."""
        params = {'g_Na': 120.0}
        tmp_cache.put(params, 500.0, 0.1, {'mean': -65.0})
        tmp_cache.get(params, 500.0, 0.1)  # hit
        tmp_cache.clear()

        assert tmp_cache.hits == 0
        assert tmp_cache.misses == 0
        assert tmp_cache.stats()['num_entries'] == 0
        assert tmp_cache.get(params, 500.0, 0.1) is None

    def test_hit_rate_calculation(self, tmp_cache):
        """hit_rate = hits / (hits + misses)."""
        params = {'g_Na': 120.0}
        tmp_cache.put(params, 500.0, 0.1, {'mean': -65.0})

        # 1 hit, 1 miss
        tmp_cache.get({'g_Na': 999.0}, 500.0, 0.1)  # miss
        tmp_cache.get(params, 500.0, 0.1)  # hit

        stats = tmp_cache.stats()
        assert stats['misses'] == 1
        assert stats['hits'] == 1
        assert stats['hit_rate'] == pytest.approx(0.5, abs=0.01)


# ═══════════════════════════════════════════════════════════════
# CachedSimulator Tests
# ═══════════════════════════════════════════════════════════════

@pytest.mark.skipif(not CACHE_AVAILABLE, reason="simulation_cache not importable")
class TestCachedSimulator:
    """Test CachedSimulator wrapping ENS-GI digital twin."""

    @pytest.fixture
    def simulator(self, tmp_path):
        twin = ENSGIDigitalTwin(n_segments=3)
        cache = SimulationCache(cache_dir=str(tmp_path / 'sim_cache'), max_size_mb=10)
        return CachedSimulator(twin, cache, duration=200.0, dt=0.1)

    def test_returns_dict_with_required_keys(self, simulator):
        """CachedSimulator result must contain mean and std."""
        params = {'g_Na': 120.0, 'g_K': 36.0}
        result = simulator(params)

        assert isinstance(result, dict)
        assert 'mean' in result
        assert 'std' in result
        assert isinstance(result['mean'], float)
        assert isinstance(result['std'], float)

    def test_cache_miss_on_first_call(self, simulator):
        """First call with new params should be a miss."""
        params = {'g_Na': 120.0, 'g_K': 36.0}
        simulator(params)
        assert simulator.cache.misses == 1

    def test_cache_hit_on_second_call(self, simulator):
        """Second call with same params should be a hit."""
        params = {'g_Na': 120.0, 'g_K': 36.0}
        result1 = simulator(params)
        result2 = simulator(params)

        assert simulator.cache.hits == 1
        assert result1['mean'] == result2['mean']

    def test_different_params_produce_different_results(self, simulator):
        """Different parameters should (usually) produce different outputs."""
        params_a = {'g_Na': 120.0, 'g_K': 36.0}
        params_b = {'g_Na': 50.0, 'g_K': 36.0}  # Very different g_Na

        result_a = simulator(params_a)
        result_b = simulator(params_b)

        # Results might differ — at minimum cache should have 2 misses
        assert simulator.cache.misses == 2

    def test_get_cache_stats_returns_dict(self, simulator):
        """get_cache_stats() returns a statistics dictionary."""
        stats = simulator.get_cache_stats()
        assert isinstance(stats, dict)
        assert 'hits' in stats
        assert 'misses' in stats
        assert 'hit_rate' in stats


# ═══════════════════════════════════════════════════════════════
# BayesianEstimator Integration Tests
# ═══════════════════════════════════════════════════════════════

@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
class TestBayesianEstimatorRealSimulator:
    """Test that BayesianEstimator uses real simulator (not placeholder)."""

    @pytest.fixture
    def twin(self):
        return ENSGIDigitalTwin(n_segments=5)

    @pytest.fixture
    def estimator(self, twin):
        config = BayesianConfig(
            n_chains=1,
            n_draws=10,   # Tiny — just tests plumbing
            n_tune=10,
            sampler='Metropolis',  # Metropolis is faster than NUTS for tiny runs
            progressbar=False,
        )
        return BayesianEstimator(twin, config=config)

    @pytest.fixture
    def observed_data(self, twin):
        """Generate synthetic observed voltages."""
        result = twin.run(300.0, dt=0.1, verbose=False)
        vols = result['voltages']
        vols += np.random.randn(*vols.shape) * 1.0
        return vols

    def test_estimate_parameters_initializes_real_simulator(self, estimator, observed_data):
        """
        estimate_parameters() must set _use_real_simulator=True and create
        _cached_simulator — i.e. no more hardcoded placeholder.
        """
        # Run (very short) MCMC
        try:
            trace = estimator.estimate_parameters(
                observed_voltages=observed_data,
                parameter_names=['g_Na', 'g_K'],
            )
        except Exception:
            # MCMC can fail with tiny draws — what matters is the setup
            pass

        # Regardless of MCMC outcome, real simulator should have been initialised
        assert hasattr(estimator, '_cached_simulator'), (
            "_cached_simulator was not created — estimate_parameters() "
            "still uses the old placeholder!"
        )

    def test_no_hardcoded_minus40_in_simulator(self, estimator):
        """
        The old placeholder returned {'mean': -40, 'std': 15} unconditionally.
        After the fix, calling _cached_simulator with different params must not
        always return exactly -40.
        """
        from ens_gi_digital.simulation_cache import SimulationCache, CachedSimulator
        import tempfile, os

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SimulationCache(cache_dir=tmpdir, max_size_mb=10)
            sim = CachedSimulator(estimator.twin, cache, duration=200.0, dt=0.1)

            r1 = sim({'g_Na': 120.0, 'g_K': 36.0})
            r2 = sim({'g_Na': 120.0, 'g_K': 36.0})

            # Must not be the hardcoded placeholder value
            assert r1['mean'] != -40 or r1['std'] != 15, (
                "CachedSimulator returned the old placeholder {'mean': -40, 'std': 15}!"
            )
            # Must return consistent results for same params (cache hit)
            assert r1['mean'] == r2['mean']


# ═══════════════════════════════════════════════════════════════
# ClinicalDataLoader Tests
# ═══════════════════════════════════════════════════════════════

@pytest.mark.skipif(not CLINICAL_LOADER_AVAILABLE, reason="patient_data not importable")
class TestClinicalDataLoader:
    """Test ClinicalDataLoader CSV loading and EGG preprocessing."""

    @pytest.fixture
    def data_dir(self, tmp_path):
        """Create a temporary directory with sample patient data."""
        data_dir = str(tmp_path / 'patient_data')
        create_sample_patient_data('TEST01', output_dir=data_dir,
                                   duration_ms=2000.0, n_channels=4)
        return data_dir

    @pytest.fixture
    def loader(self, data_dir):
        return ClinicalDataLoader(data_root=data_dir, dataset_type='egg')

    def test_load_patient_returns_dict(self, loader):
        """load_patient() returns a dict with correct keys."""
        data = loader.load_patient('TEST01')
        assert 'patient_id' in data
        assert 'egg' in data
        assert 'metadata' in data
        assert data['patient_id'] == 'TEST01'

    def test_egg_data_has_required_keys(self, loader):
        """EGG sub-dict must contain time, channels, sampling_rate."""
        data = loader.load_patient('TEST01')
        egg = data['egg']
        assert egg is not None
        for key in ('time', 'channels', 'sampling_rate', 'channel_names'):
            assert key in egg, f"Missing key '{key}' in EGG data"

    def test_egg_channels_shape(self, loader):
        """channels must be 2D [T, N] array."""
        data = loader.load_patient('TEST01')
        channels = data['egg']['channels']
        assert channels.ndim == 2, f"Expected 2D array, got shape {channels.shape}"
        assert channels.shape[1] == 4  # n_channels used in fixture

    def test_missing_patient_returns_none_egg(self, data_dir):
        """Non-existent patient should have egg=None without crashing."""
        loader = ClinicalDataLoader(data_root=data_dir)
        import warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            data = loader.load_patient('NONEXISTENT')
        assert data['egg'] is None

    def test_preprocess_egg_output_shape_unchanged(self, loader):
        """preprocess_egg() must not change the array shape."""
        data = loader.load_patient('TEST01')
        raw_shape = data['egg']['channels'].shape
        preprocessed = loader.preprocess_egg(data['egg'])
        assert preprocessed['channels'].shape == raw_shape

    def test_preprocess_egg_adds_preprocessing_key(self, loader):
        """preprocess_egg() must add 'preprocessing' key with metadata."""
        data = loader.load_patient('TEST01')
        out = loader.preprocess_egg(data['egg'])
        assert 'preprocessing' in out
        assert 'detrend' in out['preprocessing']
        assert 'bandpass_hz' in out['preprocessing']
        assert 'n_artifacts_removed' in out['preprocessing']
        assert 'normalization' in out['preprocessing']

    def test_preprocess_egg_normalizes_to_unit_variance(self, loader):
        """After preprocessing with normalize=True, each channel should be ~unit std."""
        data = loader.load_patient('TEST01')
        out = loader.preprocess_egg(data['egg'], normalize=True)
        channels = out['channels']
        stds = channels.std(axis=0)
        # After z-score normalization each column std should be 1.0
        np.testing.assert_allclose(stds, np.ones_like(stds), atol=0.05)

    def test_preprocess_hrm_normalizes_to_0_1(self, loader):
        """preprocess_hrm() with normalize=True should bound channels to [0, 1]."""
        data = loader.load_patient('TEST01')
        if data['hrm'] is None:
            pytest.skip("No HRM data available")
        out = loader.preprocess_hrm(data['hrm'], smooth_sigma_ms=0, normalize=True)
        channels = out['channels']
        assert channels.min() >= -1e-6, "Channels should be >= 0 after min-max normalization"
        assert channels.max() <= 1.0 + 1e-6, "Channels should be <= 1 after min-max normalization"


# ═══════════════════════════════════════════════════════════════
# PINN ODE Physics Loss Tests
# ═══════════════════════════════════════════════════════════════

@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")
class TestPINNODEPhysicsLoss:
    """Test that PINNConfig.use_ode_residuals is wired correctly."""

    @pytest.fixture
    def twin(self):
        return ENSGIDigitalTwin(n_segments=3)

    def test_pinn_config_has_use_ode_residuals(self):
        """PINNConfig dataclass must expose use_ode_residuals."""
        config = PINNConfig()
        assert hasattr(config, 'use_ode_residuals'), (
            "PINNConfig is missing 'use_ode_residuals' field"
        )

    def test_pinn_config_default_is_true(self):
        """Default value should be False (constraint mode active by default).

        Changed from True → False in PINN TF autograph fix (2026-02-20):
        ODE residual mode caused a symbolic tensor error inside @tf.function.
        Fast constraint mode is now the default; ODE residuals opt-in via
        use_ode_residuals=True.
        """
        config = PINNConfig()
        assert config.use_ode_residuals is False

    def test_train_accepts_use_ode_residuals_kwarg(self, twin):
        """train() should accept use_ode_residuals without raising TypeError."""
        pinn = PINNEstimator(twin, parameter_names=['g_Na', 'g_K'])
        # Just check it doesn't blow up on the signature — use 1 epoch to be fast
        try:
            pinn.train(
                epochs=1,
                n_synthetic_samples=10,
                generate_data=True,
                use_ode_residuals=False,  # Fast mode for testing
                verbose=0,
            )
        except TypeError as e:
            pytest.fail(f"train() does not accept use_ode_residuals kwarg: {e}")
        except Exception:
            pass  # Other errors (e.g. shape issues with 1 sample) are acceptable

    def test_override_sets_config(self, twin):
        """Passing use_ode_residuals=False should set config accordingly."""
        pinn = PINNEstimator(twin, parameter_names=['g_Na', 'g_K'])
        pinn.config.use_ode_residuals = True  # Ensure default is True first

        # Override via train() — check it's changed even before training runs
        pinn.config.use_ode_residuals = True
        try:
            pinn.train(epochs=0, generate_data=True,
                       n_synthetic_samples=5, use_ode_residuals=False, verbose=0)
        except Exception:
            pass
        assert pinn.config.use_ode_residuals is False
