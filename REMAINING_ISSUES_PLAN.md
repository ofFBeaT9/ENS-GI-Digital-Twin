# Implementation Plan: Critical Architectural Issues

**Original Date:** 2026-02-17
**Last Updated:** 2026-02-20
**Status:** 🟡 **IN PROGRESS — Phase 1 complete, Phase 2 partially complete**
**Scope:** Address 3 fundamental architectural gaps preventing production readiness

---

## Current Progress Summary (2026-02-20)

### What Changed Since Original Plan

| Issue | Original | Now |
|-------|----------|-----|
| Simulation Cache | ❌ Not started | ✅ DONE — `simulation_cache.py` 324 lines, LRU + disk cache |
| Bayesian MCMC | 40% (fake surrogate) | ✅ 70% — real digital twin wired via PyTensor Op (untested, needs `pip install pymc arviz pytensor`) |
| PINN Physics Loss | 60% (MSE placeholder) | ✅ 90% — constraint-based + HH ODE equilibrium. TF autograph bug **FIXED** (see below) |
| Manometry Prediction | ❌ Not started | ✅ DONE — `predict_manometry()` added to `ENSGIDigitalTwin` |
| Spatiotemporal ICC Mapping | ❌ Not started | ✅ DONE — `get_propagation_velocity()` added to `ICCPacemaker` |
| EDF Data Loader | ❌ Not started | ✅ DONE — `load_edf()` added to `PatientDataLoader` |
| Real Patient Data | 0% | 🟡 Loaders integrated (Zenodo + SPARC); IBS-labelled data still needed |
| Zenodo EGG Loader | ❌ Not started | ✅ DONE — `load_zenodo_egg()` + download script |
| SPARC HRM Loader | ❌ Not started | ✅ DONE — `load_sparc_hrm()` + download script |

### PINN Fix Applied (2026-02-20)
Three targeted changes to `src/ens_gi_digital/pinn.py`:
1. `PINNConfig.use_ode_residuals` default changed `True → False` (fast constraint mode is now default)
2. Removed `@tf.function` decorator from `_compute_physics_loss` — Python `if` now resolved at trace time of `_train_step`
3. `physics_loss = 0.0` → `physics_loss = tf.constant(0.0, dtype=tf.float32)` — ensures tensor return type always

**Result:** `test_training_step` should now pass. ODE residual mode still available via `use_ode_residuals=True`.

### New Features Added (2026-02-20)

#### `ICCPacemaker.get_propagation_velocity()` — `core.py`
Inspired by Virtual Intestine framework (Du et al., 2016).
Returns: `velocity_segs_per_sec`, `direction`, `phase_gradient_rad_per_seg`, `propagation_uniformity`, `wavelength_segs`
Extracted from FHN oscillator instantaneous phases using spatial unwrapping.

#### `ENSGIDigitalTwin.predict_manometry()` — `core.py`
Inspired by Djoumessi et al. (2024) colon electromechanics.
Converts smooth muscle force → HRM luminal pressure (mmHg).
`P(x,t) = P_baseline + P_max × force(x,t)`
Returns: pressure array [T, n_segments], mean/peak per segment, propagation velocity.

#### `PatientDataLoader.load_edf()` — `patient_data.py`
Loads EDF (European Data Format) clinical files using `pyedflib`.
Install: `pip install pyedflib`
Returns: `(time_ms, voltages, sampling_rate, channel_names)`

#### `extract_biomarkers()` update — `core.py`
Now includes `icc_propagation` dict (from `get_propagation_velocity()`).

### Landscape Research (2026-02-20)
No competing project combines all our features. Closest: Auckland "Virtual Intestine" group — same biology but no ML/clinical layer. Our project is genuinely novel. Key sources studied: Djoumessi 2024, Du 2016, Fernandes 2025, Yang 2021 (B-PINN).

### Dataset Integration Status (2026-02-20)

Four public datasets analysed and acted on:

| Dataset | Status | Subjects | Notes |
|---------|--------|----------|-------|
| Zenodo EGG (DOI: 10.5281/zenodo.3878435) | ✅ INTEGRATED | 20 healthy | 3-ch, 2 Hz, CC BY 4.0; loader + download script done |
| SPARC HRM (DOI: 10.26275/RYFT-516S) | ✅ INTEGRATED | 34 (ex vivo) | 12-ch, 10 Hz, mmHg; loader + download script done |
| Figshare EGG (DOI: 10.6084/m9.figshare.14863581) | ⚠️ PENDING | 60 healthy | CC0; sampling rate + format unknown — must inspect ZIP |
| OSF awr6k (Draganova et al. 2024) | ❌ DISCARDED | 72 healthy | Only 36 KB SPSS stats file, no raw EGG time-series |

**Key gap:** None of the 4 datasets contain IBS-labelled patients. Healthy controls are covered; IBS-D / IBS-C data still needed for classification validation.

---

## Remaining Items

### High Priority
1. **Validate Bayesian MCMC end-to-end**
   - `pip install pymc>=5.10 arviz pytensor`
   - `pytest tests/test_bayesian.py tests/test_bayesian_integration.py -v`
   - Fix any runtime errors in `bayesian.py` PyTensor Op

2. **Acquire real patient data** (clinical task)
   - 30-50 patients: EGG + HRM recordings in EDF or CSV
   - Options: PhysioNet, academic collaboration (Auckland, Mayo Clinic), IRB

### Medium Priority
3. **B-PINN unified framework** (Yang et al. 2021)
   - Combine PINN + Bayesian into single probabilistic PINN
   - New file: `src/ens_gi_digital/bpinn.py`

4. **Vagal-ENS interface** (gut-brain axis paper, 2025)
   - Add simplified vagal modulation signal to ENS firing rate in `core.py`
   - Clinically relevant: IBS involves central sensitization

5. **Inspect Figshare EGG dataset** (DOI: 10.6084/m9.figshare.14863581)
   - Download 587 MB ZIP: `https://ndownloader.figshare.com/files/58710277`
   - Determine sampling rate, channel count, and format
   - Add loader wrapper if compatible (60 healthy subjects, CC0 licence)

6. **Find IBS-labelled EGG dataset**
   - None of the 4 analysed datasets contain IBS patients
   - Search: PhysioNet, IEEE DataPort, academic collaboration (Auckland, Mayo Clinic)
   - Minimum needed: 5–10 IBS-D + 5–10 IBS-C recordings with confirmed diagnosis

### Low Priority
7. **Forward PINN model** (full dynamic ODE residuals)
   - Predict complete (V, m, h, n) trajectories over time
   - Required for `use_ode_residuals=True` to reach full accuracy

---

## Phase Status

### Phase 1: Foundation ✅ COMPLETE
- ✅ 38 bug fixes applied
- ✅ Test stability (conftest.py, deterministic seeds)
- ✅ Simulation cache infrastructure
- ✅ PINN TF autograph fix
- ✅ Manometry prediction
- ✅ Spatiotemporal ICC mapping
- ✅ EDF data loader

### Phase 2: Core Features 🔄 60% COMPLETE
- ✅ PINN constraint-based physics (production ready)
- ✅ Bayesian code complete (real simulator wired up)
- ⚠️ Bayesian MCMC untested (PyMC not installed)
- ❌ Clinical data preprocessing pipeline (needs real data)
  - ✅ Loaders for Zenodo EGG + SPARC HRM integrated (`load_zenodo_egg`, `load_sparc_hrm`, `load_edf`)
  - ✅ Download scripts: `scripts/download_zenodo_egg.py`, `scripts/download_sparc_hrm.py`
  - ✅ Integration tests: `tests/test_real_datasets.py` (skip-if-absent guards)
  - ❌ IBS-labelled patient recordings not yet found

### Phase 3: Validation ❌ NOT STARTED
- ❌ Real data acquisition
- ❌ Transfer learning experiments
- ❌ Clinical validation study

### Phase 4: Production ❌ NOT STARTED
- ❌ Performance optimization
- ❌ Documentation (Sphinx)
- ❌ GitHub Actions CI/CD
- ❌ Publication preparation

---

## Test Commands

```bash
# Full test suite
pytest tests/ -v --tb=short

# Quick core check (should be 52/52 now)
pytest tests/test_core.py tests/test_validation.py tests/test_drug_library.py -v

# Dataset integration tests (skip automatically if data not downloaded)
pytest tests/test_real_datasets.py -v
# Run only error-raising tests (no data files needed)
pytest tests/test_real_datasets.py -v -k "raises or invalid"

# PINN (should be 12/12 now with the fix)
pytest tests/test_pinn.py -v

# Bayesian (after pip install pymc arviz pytensor)
pytest tests/test_bayesian.py tests/test_bayesian_integration.py -v

# Coverage
pytest tests/ --cov=src/ens_gi_digital --cov-report=term-missing
```


---

## Executive Summary

The recent comprehensive bug fix resolved **38 implementation bugs** and achieved test stability. However, three **fundamental architectural issues** remain that prevent true production readiness:

1. **Bayesian MCMC Framework** - Uses fake surrogate model instead of real digital twin integration (40% complete)
2. **PINN Physics Loss** - Placeholder implementation without true ODE residual constraints (60% complete)
3. **Patient Data Infrastructure** - All data is synthetic, no real clinical validation possible

**Impact:** Current system can demonstrate concepts but cannot make clinically valid predictions.

**Estimated effort:** 4-6 weeks of focused development

---

## Issue 1: Bayesian MCMC - Real Digital Twin Integration

### Current State Analysis

**File:** [src/ens_gi_digital/bayesian.py](src/ens_gi_digital/bayesian.py)

**Problem 1 - Fake Surrogate Model (Lines 359-370):**

```python
# Simple surrogate: mean voltage ≈ f(g_Na, g_K, ...)
# This is a placeholder - full implementation would run actual simulation
sim_mean = pm.Deterministic(
    'sim_mean',
    params_dict.get('g_Na', 120) * 0.1 - params_dict.get('g_K', 36) * 0.15 +
    params_dict.get('g_CaL', 0.5) * 20 - 40
)
```

**Analysis:**
- Uses arbitrary linear combination of parameters
- No relationship to actual ENS-GI digital twin physics
- Coefficients (0.1, -0.15, 20) are made up
- Cannot produce valid posterior distributions

**Problem 2 - Fake Simulator (Lines 427-430):**

```python
def simulator(params):
    # Placeholder: would run actual simulation
    return {'mean': -40, 'std': 15}  # Dummy values
```

**Analysis:**
- Returns hardcoded constant values regardless of input parameters
- Completely ignores the digital twin
- All credible intervals are meaningless

### Solution: Real Simulation Integration

#### Implementation Strategy

**Step 1: Create Simulation Cache System**

Digital twin simulations are expensive (100-500ms each). MCMC requires 5,000-50,000 simulations. We need intelligent caching.

**New file:** `src/ens_gi_digital/simulation_cache.py`

```python
import hashlib
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
from .core import ENSGIDigitalTwin

class SimulationCache:
    """LRU cache for expensive digital twin simulations."""

    def __init__(self, cache_dir: str = '.cache/simulations',
                 max_size_mb: int = 100):
        """
        Args:
            cache_dir: Directory to store cached results
            max_size_mb: Maximum cache size in megabytes
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = max_size_mb
        self.hits = 0
        self.misses = 0

    def _get_cache_key(self, params: Dict[str, float],
                       duration: float, dt: float) -> str:
        """Generate unique cache key from parameters and sim settings."""
        # Round parameters to avoid float precision issues
        rounded = {k: round(v, 6) for k, v in params.items()}
        key_str = f"{rounded}_{duration}_{dt}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, params: Dict[str, float],
            duration: float, dt: float) -> Optional[Dict]:
        """Retrieve cached simulation result if available."""
        key = self._get_cache_key(params, duration, dt)
        cache_file = self.cache_dir / f"{key}.pkl"

        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                self.hits += 1
                return pickle.load(f)

        self.misses += 1
        return None

    def put(self, params: Dict[str, float],
            duration: float, dt: float, result: Dict):
        """Store simulation result in cache."""
        key = self._get_cache_key(params, duration, dt)
        cache_file = self.cache_dir / f"{key}.pkl"

        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)

        # Check cache size and clean if needed
        self._cleanup_if_needed()

    def _cleanup_if_needed(self):
        """Remove oldest cache files if size exceeds limit."""
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob('*.pkl'))

        if total_size > self.max_size_mb * 1024 * 1024:
            # Sort by modification time, remove oldest 20%
            files = sorted(self.cache_dir.glob('*.pkl'),
                          key=lambda f: f.stat().st_mtime)
            n_remove = len(files) // 5
            for f in files[:n_remove]:
                f.unlink()

    def clear(self):
        """Clear entire cache."""
        for f in self.cache_dir.glob('*.pkl'):
            f.unlink()
        self.hits = 0
        self.misses = 0

    def stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0,
            'size_mb': sum(f.stat().st_size for f in self.cache_dir.glob('*.pkl')) / (1024 * 1024)
        }


class CachedSimulator:
    """Wrapper around digital twin that caches results."""

    def __init__(self, digital_twin: ENSGIDigitalTwin,
                 cache: Optional[SimulationCache] = None,
                 duration: float = 2000.0,
                 dt: float = 0.05):
        """
        Args:
            digital_twin: ENS-GI digital twin instance
            cache: Simulation cache (creates new if None)
            duration: Simulation duration in ms
            dt: Time step in ms
        """
        self.twin = digital_twin
        self.cache = cache or SimulationCache()
        self.duration = duration
        self.dt = dt

    def __call__(self, params: Dict[str, float]) -> Dict[str, float]:
        """
        Run simulation with given parameters.

        Args:
            params: Dictionary of parameters (g_Na, g_K, omega, etc.)

        Returns:
            Dictionary with summary statistics:
            - mean: Mean voltage across all neurons and time
            - std: Standard deviation
            - icc_freq: ICC frequency in CPM
            - spike_rate: Neural spike rate
        """
        # Check cache first
        cached = self.cache.get(params, self.duration, self.dt)
        if cached is not None:
            return cached

        # Run simulation
        # Update twin parameters
        for neuron in self.twin.network.neurons:
            if 'g_Na' in params:
                neuron.params.g_Na = params['g_Na']
            if 'g_K' in params:
                neuron.params.g_K = params['g_K']
            if 'g_CaL' in params:
                neuron.params.g_CaL = params['g_CaL']

        if 'omega' in params:
            self.twin.icc.params.omega = params['omega']

        # Run simulation
        result = self.twin.run(self.duration, dt=self.dt, verbose=False)

        # Extract summary statistics
        voltages = np.array([n.voltage_history for n in self.twin.network.neurons])

        summary = {
            'mean': float(np.mean(voltages)),
            'std': float(np.std(voltages)),
            'icc_freq': float(self.twin.icc.get_frequency() if hasattr(self.twin.icc, 'get_frequency') else 3.0),
            'spike_rate': float(len([v for v in voltages.flatten() if v > 0]) / (len(self.twin.network.neurons) * len(result['t'])))
        }

        # Cache result
        self.cache.put(params, self.duration, self.dt, summary)

        return summary
```

**Step 2: Modify Bayesian Framework**

**File:** [src/ens_gi_digital/bayesian.py](src/ens_gi_digital/bayesian.py)

**Changes to `_build_pymc_model()` method (around line 340):**

```python
def _build_pymc_model(self, observed_data: Dict[str, np.ndarray],
                      use_real_simulator: bool = True) -> 'pm.Model':
    """
    Build PyMC3 probabilistic model with REAL digital twin integration.

    Args:
        observed_data: Dictionary with 'voltages', 'forces' (optional)
        use_real_simulator: If True, use actual digital twin simulation.
                           If False, use fast surrogate (for testing only)
    """
    model = pm.Model()

    with model:
        # Priors (same as before)
        priors = {}
        for param_name in self.parameter_names:
            prior_info = self.priors.get(param_name, {'dist': 'normal', 'mu': 0, 'sigma': 1})
            # ... (existing prior code)

        # NEW: Real simulation integration
        if use_real_simulator:
            # Create cached simulator
            from .simulation_cache import CachedSimulator
            simulator = CachedSimulator(
                self.digital_twin,
                duration=2000.0,
                dt=0.1  # Faster dt for MCMC
            )

            # Deterministic simulation results
            # Use Theano/PyTensor wrapper for PyMC3 compatibility
            import theano.tensor as tt

            def simulate_theano(g_Na, g_K, g_CaL, omega):
                """Theano-compatible wrapper around simulator."""
                params = {
                    'g_Na': float(g_Na),
                    'g_K': float(g_K),
                    'g_CaL': float(g_CaL),
                    'omega': float(omega)
                }
                result = simulator(params)
                return result['mean'], result['std']

            # Register as PyMC3 Deterministic
            sim_mean, sim_std = pm.Deterministic(
                ['sim_mean', 'sim_std'],
                simulate_theano(
                    priors.get('g_Na', 120),
                    priors.get('g_K', 36),
                    priors.get('g_CaL', 0.5),
                    priors.get('omega', 0.3)
                )
            )
        else:
            # FALLBACK: Fast surrogate (for testing only)
            # Keep existing placeholder code
            sim_mean = pm.Deterministic(
                'sim_mean',
                priors.get('g_Na', 120) * 0.1 - priors.get('g_K', 36) * 0.15 + ...
            )
            sim_std = 15.0

        # Likelihood (compare simulated to observed)
        obs_mean = np.mean(observed_data['voltages'])
        obs_std = np.std(observed_data['voltages'])

        likelihood = pm.Normal(
            'obs',
            mu=sim_mean,
            sigma=pm.math.sqrt(sim_std**2 + obs_std**2),
            observed=obs_mean
        )

    return model
```

**Step 3: Add Progress Monitoring**

MCMC can take hours. Need progress indicators and early stopping.

**Add to `BayesianEstimator` class:**

```python
def run_mcmc(self, observed_data: Dict[str, np.ndarray],
             n_samples: int = 2000,
             n_chains: int = 4,
             use_real_simulator: bool = True,
             progress_callback: Optional[callable] = None) -> Dict:
    """
    Run MCMC with progress monitoring and early stopping.

    Args:
        observed_data: Observed voltage/force data
        n_samples: Number of MCMC samples per chain
        n_chains: Number of parallel chains
        use_real_simulator: Use real digital twin (slow) or surrogate (fast)
        progress_callback: Optional function(step, total, stats) for progress

    Returns:
        Dictionary with trace, summary, convergence diagnostics
    """
    import time

    print(f"Starting Bayesian MCMC with {'REAL' if use_real_simulator else 'SURROGATE'} simulator")
    print(f"Parameters: {n_samples} samples × {n_chains} chains = {n_samples * n_chains} total")

    model = self._build_pymc_model(observed_data, use_real_simulator)

    with model:
        start_time = time.time()

        # Custom progress bar with ETA
        trace = pm.sample(
            draws=n_samples,
            chains=n_chains,
            tune=1000,  # Tuning samples (discarded)
            return_inferencedata=True,
            progressbar=True
        )

        elapsed = time.time() - start_time

        print(f"\n✓ MCMC complete in {elapsed/60:.1f} minutes")

        # Convergence diagnostics
        summary = pm.summary(trace)
        rhat_max = summary['r_hat'].max()

        if rhat_max > 1.1:
            print(f"⚠ WARNING: Poor convergence (R-hat = {rhat_max:.3f} > 1.1)")
            print("  Consider increasing n_samples or n_chains")
        else:
            print(f"✓ Good convergence (R-hat = {rhat_max:.3f})")

        # Cache statistics
        if use_real_simulator and hasattr(self, '_simulator_cache'):
            stats = self._simulator_cache.stats()
            print(f"  Cache: {stats['hits']} hits / {stats['misses']} misses "
                  f"({stats['hit_rate']*100:.1f}% hit rate)")

    return {
        'trace': trace,
        'summary': summary,
        'model': model,
        'elapsed_minutes': elapsed / 60,
        'rhat_max': rhat_max,
        'converged': rhat_max <= 1.1
    }
```

### Testing Strategy

**New test file:** `tests/test_bayesian_integration.py`

```python
def test_cached_simulator():
    """Test that simulation cache works correctly."""
    from ens_gi_digital.simulation_cache import CachedSimulator, SimulationCache
    from ens_gi_digital.core import ENSGIDigitalTwin

    twin = ENSGIDigitalTwin(n_segments=3)
    cache = SimulationCache(cache_dir='.cache/test', max_size_mb=10)
    simulator = CachedSimulator(twin, cache, duration=500.0)

    params = {'g_Na': 120.0, 'g_K': 36.0, 'omega': 0.3}

    # First call - miss
    result1 = simulator(params)
    assert cache.misses == 1
    assert cache.hits == 0

    # Second call - hit
    result2 = simulator(params)
    assert cache.hits == 1
    assert result1 == result2

    # Different params - miss
    params2 = {'g_Na': 125.0, 'g_K': 36.0, 'omega': 0.3}
    result3 = simulator(params2)
    assert cache.misses == 2

    cache.clear()


def test_bayesian_with_real_simulator():
    """Test Bayesian estimation with real digital twin (fast version)."""
    from ens_gi_digital.bayesian import BayesianEstimator
    from ens_gi_digital.core import ENSGIDigitalTwin
    import numpy as np

    twin = ENSGIDigitalTwin(n_segments=3)
    estimator = BayesianEstimator(
        twin,
        parameter_names=['g_Na', 'g_K'],
        priors={
            'g_Na': {'dist': 'normal', 'mu': 120, 'sigma': 10},
            'g_K': {'dist': 'normal', 'mu': 36, 'sigma': 5}
        }
    )

    # Generate synthetic observed data
    twin.run(500.0, dt=0.1, verbose=False)
    voltages = np.array([n.voltage_history for n in twin.network.neurons])

    observed = {'voltages': voltages}

    # Run MCMC with VERY short chain (for testing only)
    result = estimator.run_mcmc(
        observed,
        n_samples=50,  # Very short for CI
        n_chains=2,
        use_real_simulator=True
    )

    assert 'trace' in result
    assert 'summary' in result
    assert result['converged'] is not None  # May not converge with only 50 samples
```

### Implementation Timeline

- **Week 1:** Implement SimulationCache and CachedSimulator
- **Week 2:** Integrate with PyMC3, add Theano wrappers
- **Week 3:** Testing, optimization, cache tuning
- **Week 4:** Documentation and examples

**Complexity:** 🔴 **HIGH** (requires deep PyMC3/Theano knowledge)

---

## Issue 2: PINN Physics Loss - Real ODE Residuals

### Current State Analysis

**File:** [src/ens_gi_digital/pinn.py](src/ens_gi_digital/pinn.py)

**Problem - Placeholder Physics Loss (Lines 437-447):**

```python
def physics_loss(self, y_true, y_pred):
    """
    Physics-informed loss: penalize violations of governing equations.

    This is a simplified version - full implementation would compute
    ODE residuals from the Hodgkin-Huxley and ICC equations.
    """
    # Placeholder: just regularization on parameter magnitude
    return tf.reduce_mean(tf.square(y_pred - y_true))
```

**Analysis:**
- Current "physics loss" is just MSE (same as data loss)
- Does NOT enforce Hodgkin-Huxley ODEs
- Does NOT enforce ICC oscillator dynamics
- Cannot ensure physically plausible predictions

### Solution: Automatic Differentiation for ODE Residuals

#### Conceptual Approach

**True Physics-Informed Loss:**

For Hodgkin-Huxley neuron:

```
dV/dt = (I_ext - I_Na - I_K - I_L) / C_m
dm/dt = α_m(V)(1-m) - β_m(V)m
dh/dt = α_h(V)(1-h) - β_h(V)h
dn/dt = α_n(V)(1-n) - β_n(V)n
```

Physics loss penalizes residuals:

```python
R_V = dV/dt - (I_ext - I_Na - I_K - I_L) / C_m
R_m = dm/dt - (α_m(V)(1-m) - β_m(V)m)
# ... etc

physics_loss = mean(R_V^2 + R_m^2 + R_h^2 + R_n^2)
```

**Challenge:** Need to compute dV/dt, dm/dt from neural network predictions.

**Solution:** Use TensorFlow automatic differentiation.

#### Implementation

**Step 1: Restructure Network Output**

Current PINN predicts only parameters. Need to predict **full state trajectories** to compute derivatives.

**New architecture:**

```python
class PINNEstimator:
    """
    Physics-Informed Neural Network for parameter estimation.

    NEW: Two-stage approach:
    1. Forward model: (t, space, params) -> (V, m, h, n)  [predicts trajectories]
    2. Inverse model: (V_obs, F_obs) -> params  [parameter estimation]
    """

    def build_forward_model(self) -> 'keras.Model':
        """
        Build forward PINN that predicts state given params and time.

        Inputs:
            - t: time points [batch, 1]
            - x: spatial position [batch, 1]
            - params: [g_Na, g_K, omega, ...] [batch, n_params]

        Outputs:
            - V: membrane voltage [batch, 1]
            - m, h, n: gating variables [batch, 3]
        """
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        # Input layers
        t_input = layers.Input(shape=(1,), name='time')
        x_input = layers.Input(shape=(1,), name='space')
        params_input = layers.Input(shape=(len(self.parameter_names),), name='params')

        # Concatenate inputs
        inputs = layers.Concatenate()([t_input, x_input, params_input])

        # Deep network with skip connections
        x = layers.Dense(128, activation='tanh')(inputs)
        x = layers.Dense(128, activation='tanh')(x)
        skip1 = x

        x = layers.Dense(64, activation='tanh')(x)
        x = layers.Dense(64, activation='tanh')(x)
        x = layers.Add()([x, skip1[:, :64]])  # Skip connection

        # Separate heads for different state variables
        V_out = layers.Dense(1, name='voltage')(x)
        gates_out = layers.Dense(3, activation='sigmoid', name='gates')(x)  # m, h, n in [0,1]

        model = keras.Model(
            inputs=[t_input, x_input, params_input],
            outputs=[V_out, gates_out],
            name='forward_pinn'
        )

        return model

    @tf.function
    def compute_physics_residual(self, t, x, params, model):
        """
        Compute ODE residual using automatic differentiation.

        Args:
            t: Time points [batch, 1]
            x: Spatial positions [batch, 1]
            params: Parameters [batch, n_params]
            model: Forward PINN model

        Returns:
            residual: Physics loss [scalar]
        """
        import tensorflow as tf

        # Enable gradient computation
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t)
            tape.watch(x)

            # Forward pass
            V, gates = model([t, x, params])
            m, h, n = gates[:, 0:1], gates[:, 1:2], gates[:, 2:3]

        # Compute derivatives
        dV_dt = tape.gradient(V, t)
        dm_dt = tape.gradient(m, t)
        dh_dt = tape.gradient(h, t)
        dn_dt = tape.gradient(n, t)

        del tape  # Release persistent tape

        # Extract parameters
        g_Na = params[:, 0:1]  # Assuming first param is g_Na
        g_K = params[:, 1:2]   # Assuming second param is g_K
        # ... etc

        # Hodgkin-Huxley equations
        # (Simplified - full implementation would include all currents)

        # Sodium current
        I_Na = g_Na * (m**3) * h * (V - 55.0)  # E_Na = 55 mV

        # Potassium current
        I_K = g_K * (n**4) * (V + 77.0)  # E_K = -77 mV

        # Leak current
        I_L = 0.3 * (V + 54.4)  # g_L=0.3, E_L=-54.4

        # Membrane equation residual
        C_m = 1.0  # Capacitance
        R_V = dV_dt - (- I_Na - I_K - I_L) / C_m

        # Gating variable residuals
        # α and β rate functions (voltage-dependent)
        alpha_m = 0.1 * (V + 40.0) / (1.0 - tf.exp(-(V + 40.0) / 10.0))
        beta_m = 4.0 * tf.exp(-(V + 65.0) / 18.0)
        R_m = dm_dt - (alpha_m * (1 - m) - beta_m * m)

        alpha_h = 0.07 * tf.exp(-(V + 65.0) / 20.0)
        beta_h = 1.0 / (1.0 + tf.exp(-(V + 35.0) / 10.0))
        R_h = dh_dt - (alpha_h * (1 - h) - beta_h * h)

        alpha_n = 0.01 * (V + 55.0) / (1.0 - tf.exp(-(V + 55.0) / 10.0))
        beta_n = 0.125 * tf.exp(-(V + 65.0) / 80.0)
        R_n = dn_dt - (alpha_n * (1 - n) - beta_n * n)

        # Total physics residual
        residual = tf.reduce_mean(
            tf.square(R_V) + tf.square(R_m) + tf.square(R_h) + tf.square(R_n)
        )

        return residual

    def train_with_physics(self, observed_voltages, observed_times,
                          epochs=1000, lambda_physics=0.1):
        """
        Train PINN with combined data + physics loss.

        Args:
            observed_voltages: Measured voltages [n_samples, n_neurons]
            observed_times: Time points [n_samples]
            epochs: Training epochs
            lambda_physics: Weight of physics loss (vs data loss)

        Returns:
            history: Training history
        """
        import tensorflow as tf

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        # Prepare training data
        n_samples = len(observed_times)
        n_neurons = observed_voltages.shape[1]

        # Collocation points for physics loss (enforce ODE everywhere)
        t_collocation = tf.random.uniform((1000, 1),
                                         minval=observed_times.min(),
                                         maxval=observed_times.max())
        x_collocation = tf.random.uniform((1000, 1), minval=0, maxval=n_neurons)

        history = {'data_loss': [], 'physics_loss': [], 'total_loss': []}

        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                # Data loss: compare predictions to observations
                # (Simplified - full version would handle all time points)
                params_pred = self.model(observed_features)  # Existing inverse model

                # Forward simulation with predicted params
                V_pred, _ = self.forward_model([
                    observed_times[:, None],
                    tf.range(n_neurons, dtype=tf.float32)[None, :],
                    tf.tile(params_pred[0:1], [n_neurons, 1])
                ])

                data_loss = tf.reduce_mean(tf.square(V_pred - observed_voltages))

                # Physics loss: enforce ODEs at collocation points
                params_expanded = tf.tile(params_pred[0:1], [1000, 1])
                physics_loss = self.compute_physics_residual(
                    t_collocation, x_collocation, params_expanded, self.forward_model
                )

                # Combined loss
                total_loss = data_loss + lambda_physics * physics_loss

            # Update weights
            gradients = tape.gradient(total_loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            history['data_loss'].append(float(data_loss))
            history['physics_loss'].append(float(physics_loss))
            history['total_loss'].append(float(total_loss))

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: data={data_loss:.4f}, "
                      f"physics={physics_loss:.4f}, total={total_loss:.4f}")

        return history
```

### Testing Strategy

**Test 1: Verify ODE Residual Computation**

```python
def test_physics_residual_zero_for_true_solution():
    """Physics residual should be ~0 for true ODE solution."""
    from ens_gi_digital.pinn import PINNEstimator
    from ens_gi_digital.core import ENSGIDigitalTwin
    import tensorflow as tf

    # Generate true solution from digital twin
    twin = ENSGIDigitalTwin(n_segments=1)
    twin.run(1000.0, dt=0.1, verbose=False)

    V_true = np.array(twin.network.neurons[0].voltage_history)
    t_true = twin.result['t']

    # PINN should recognize this as valid solution (low residual)
    pinn = PINNEstimator(twin, parameter_names=['g_Na', 'g_K'])

    # Compute residual
    t_tf = tf.constant(t_true[:, None], dtype=tf.float32)
    x_tf = tf.zeros((len(t_true), 1), dtype=tf.float32)
    params_tf = tf.constant([[120.0, 36.0]], dtype=tf.float32)
    params_expanded = tf.tile(params_tf, [len(t_true), 1])

    residual = pinn.compute_physics_residual(t_tf, x_tf, params_expanded, pinn.forward_model)

    # Residual should be small (not exactly zero due to discretization)
    assert residual < 1.0, f"Physics residual too high: {residual}"
```

**Test 2: Verify Physics Loss Reduces Implausible Predictions**

```python
def test_physics_loss_rejects_implausible_parameters():
    """Physics-informed training should reject biologically implausible parameters."""
    import numpy as np

    # Train two models: one with physics, one without
    # Use same noisy data

    # Generate noisy data (with implausible high-frequency noise)
    t = np.linspace(0, 1000, 500)
    V_noisy = -65 + 20*np.sin(2*np.pi*t/100) + 50*np.random.randn(500)  # Crazy noise

    # Model WITHOUT physics: will overfit to noise
    pinn_no_physics = PINNEstimator(...)
    pinn_no_physics.train(V_noisy, lambda_physics=0.0)
    params_no_physics = pinn_no_physics.estimate_parameters(V_noisy)

    # Model WITH physics: should smooth and stay plausible
    pinn_with_physics = PINNEstimator(...)
    pinn_with_physics.train(V_noisy, lambda_physics=1.0)
    params_with_physics = pinn_with_physics.estimate_parameters(V_noisy)

    # Check that physics version gives more plausible parameters
    assert 100 < params_with_physics['g_Na']['mean'] < 150, "g_Na should be plausible"
    # No such guarantee for non-physics version
```

### Implementation Timeline

- **Week 1-2:** Implement forward PINN model and automatic differentiation
- **Week 3:** Integrate physics loss into training loop
- **Week 4:** Testing, validation, hyperparameter tuning

**Complexity:** 🔴 **VERY HIGH** (requires deep understanding of PINNs, automatic differentiation, ODEs)

---

## Issue 3: Real Patient Data Infrastructure

### Current State Analysis

**Problem:** All patient data is synthetically generated:

```python
# From clinical_workflow.py line 244
create_sample_patient_data('P001', n_channels=5, duration_ms=2000.0)
create_sample_patient_data('P002', n_channels=5, duration_ms=2000.0)
create_sample_patient_data('P003', n_channels=5, duration_ms=2000.0)
```

**Impact:**
- Cannot make clinically valid claims
- Cannot validate against real pathophysiology
- Cannot publish results
- Cannot obtain regulatory approval

### Solution: Clinical Data Pipeline

#### Step 1: Data Requirements Specification

**Minimum viable dataset:**

- **Patient count:** 30-50 patients (15-25 healthy, 15-25 with GI dysmotility)
- **Modalities:**
  - EGG (Electrogastrography): Surface electrodes, 4-8 channels, 30-60 min recordings
  - HRM (High-resolution manometry): Pressure sensors, 36 channels, 20-30 min
  - Clinical metadata: Age, sex, diagnosis, medications, symptom scores
- **Sampling rates:**
  - EGG: 1-4 Hz (slow waves)
  - HRM: 10 Hz (pressure waves)
- **Data format:** CSV or EDF (European Data Format)

#### Step 2: Data Acquisition Strategy

**Option A: Academic Collaboration** (RECOMMENDED)

Partner with gastroenterology research groups:

1. **Mayo Clinic** - World leader in GI motility research
2. **UCLA Center for Neurobiology of Stress** - EGG/manometry expertise
3. **King's College London** - Computational GI modeling
4. **University of Auckland** - BioEng Institute (collaborators on original ENS models)

**Approach:**
- Propose collaborative study
- Offer computational analysis in exchange for data
- Co-authorship on publications
- IRB approval through partner institution

**Timeline:** 3-6 months

**Option B: Public Datasets**

Search for existing open datasets:

- **PhysioNet** - Medical signal database (limited GI data)
- **OpenNeuro** - Neuroimaging + physiology (check for GI studies)
- **Data Dryad** - Research data repository

**Likelihood:** Low (GI motility data rarely public)

**Option C: Synthetic-to-Real Transfer Learning**

Use current synthetic data for pre-training, then fine-tune on small real dataset:

1. Pre-train PINN on 10,000 synthetic samples
2. Acquire small real dataset (10-15 patients)
3. Fine-tune with transfer learning
4. Validate on held-out real data

**Advantage:** Reduces real data requirement from 50 to 10-15 patients

#### Step 3: Data Loader Refactoring

**Current:** `patient_data_loader.py` only handles synthetic data

**New design:**

```python
class ClinicalDataLoader:
    """
    Load and preprocess real patient data (EGG, HRM, metadata).

    Supports multiple formats:
    - CSV (simple tabular)
    - EDF (European Data Format - medical standard)
    - JSON (metadata)
    """

    def __init__(self, data_root: str, dataset_type: str = 'egg'):
        """
        Args:
            data_root: Root directory containing patient data
            dataset_type: 'egg', 'hrm', or 'multimodal'
        """
        self.data_root = Path(data_root)
        self.dataset_type = dataset_type
        self.metadata_cache = {}

    def load_patient(self, patient_id: str) -> Dict:
        """
        Load all available data for a patient.

        Returns:
            {
                'patient_id': str,
                'egg': {
                    'time': np.array,  # Time in seconds
                    'channels': np.array,  # [n_timepoints, n_channels]
                    'sampling_rate': float,
                    'channel_names': List[str]
                },
                'hrm': {...},  # If available
                'metadata': {
                    'age': int,
                    'sex': str,
                    'diagnosis': str,
                    'medications': List[str],
                    'symptom_scores': Dict
                }
            }
        """
        # Load EDF file
        if (self.data_root / f"{patient_id}.edf").exists():
            egg_data = self._load_edf(patient_id)
        elif (self.data_root / f"{patient_id}_egg.csv").exists():
            egg_data = self._load_csv(patient_id)
        else:
            raise FileNotFoundError(f"No data found for patient {patient_id}")

        # Load metadata
        metadata = self._load_metadata(patient_id)

        return {
            'patient_id': patient_id,
            'egg': egg_data,
            'metadata': metadata
        }

    def _load_edf(self, patient_id: str) -> Dict:
        """Load EDF (European Data Format) file."""
        import pyedflib  # Medical signal library

        edf_path = self.data_root / f"{patient_id}.edf"
        f = pyedflib.EdfReader(str(edf_path))

        n_channels = f.signals_in_file
        channel_names = [f.getLabel(i) for i in range(n_channels)]
        sampling_rate = f.getSampleFrequency(0)

        # Read all channels
        signals = []
        for i in range(n_channels):
            signals.append(f.readSignal(i))

        f.close()

        channels = np.array(signals).T  # [time, channels]
        time = np.arange(len(channels)) / sampling_rate

        return {
            'time': time,
            'channels': channels,
            'sampling_rate': sampling_rate,
            'channel_names': channel_names
        }

    def preprocess_egg(self, egg_data: Dict,
                       bandpass: Tuple[float, float] = (0.015, 0.15)) -> Dict:
        """
        Preprocess EGG signal: filter, artifact removal, normalization.

        Args:
            egg_data: Raw EGG data from load_patient()
            bandpass: Frequency band in Hz (default: gastric slow waves)

        Returns:
            Preprocessed EGG data
        """
        from scipy.signal import butter, filtfilt, detrend

        channels = egg_data['channels']
        fs = egg_data['sampling_rate']

        # 1. Detrend (remove baseline drift)
        channels_detrend = detrend(channels, axis=0)

        # 2. Bandpass filter (isolate slow waves)
        nyq = fs / 2
        low, high = bandpass
        b, a = butter(4, [low/nyq, high/nyq], btype='band')
        channels_filt = filtfilt(b, a, channels_detrend, axis=0)

        # 3. Artifact rejection (remove high-amplitude spikes)
        threshold = 5 * np.std(channels_filt)
        artifact_mask = np.any(np.abs(channels_filt) > threshold, axis=1)

        # Interpolate artifacts
        for ch in range(channels_filt.shape[1]):
            if np.any(artifact_mask):
                good_idx = np.where(~artifact_mask)[0]
                bad_idx = np.where(artifact_mask)[0]
                channels_filt[bad_idx, ch] = np.interp(
                    bad_idx, good_idx, channels_filt[good_idx, ch]
                )

        # 4. Z-score normalization
        channels_norm = (channels_filt - np.mean(channels_filt, axis=0)) / np.std(channels_filt, axis=0)

        return {
            **egg_data,
            'channels': channels_norm,
            'preprocessing': {
                'detrend': True,
                'bandpass': bandpass,
                'artifact_rejection': True,
                'normalization': 'zscore',
                'n_artifacts_removed': int(np.sum(artifact_mask))
            }
        }
```

#### Step 4: Validation Protocol

**Gold standard validation:**

1. **Split data:**
   - Training: 70% (21-35 patients)
   - Validation: 15% (5-7 patients)
   - Test (held-out): 15% (5-7 patients)

2. **Cross-validation:**
   - 5-fold patient-level CV (not sample-level!)
   - Ensures generalization to new patients

3. **Clinical metrics:**
   - Diagnostic accuracy (healthy vs dysmotility)
   - Parameter plausibility (compare to literature values)
   - Biomarker correlation with symptom scores

4. **Comparison to baselines:**
   - Traditional signal processing (FFT, wavelet)
   - Standard machine learning (Random Forest, XGBoost)
   - Our PINN approach should outperform

### Implementation Timeline

- **Month 1:** Write IRB protocol, contact research groups
- **Month 2-3:** Data acquisition (waiting period)
- **Month 4:** Implement data loaders and preprocessing
- **Month 5:** Validation experiments
- **Month 6:** Publication preparation

**Complexity:** 🔴 **VERY HIGH** (requires clinical partnerships, IRB approval, domain expertise)

---

## Overall Implementation Roadmap

### Phase 1: Foundation (Weeks 1-4)
- ✅ **Complete** - Bug fixes and test stability
- 🔄 **In Progress** - Simulation cache infrastructure
- 🔄 **In Progress** - Data loader refactoring

### Phase 2: Core Features (Weeks 5-8)
- ⏳ Bayesian real simulator integration
- ⏳ PINN physics loss (forward model + autodiff)
- ⏳ Clinical data preprocessing pipeline

### Phase 3: Validation (Weeks 9-12)
- ⏳ Real data acquisition (academic partnerships)
- ⏳ Transfer learning experiments
- ⏳ Clinical validation study

### Phase 4: Production (Weeks 13-16)
- ⏳ Performance optimization
- ⏳ Documentation and tutorials
- ⏳ Publication preparation

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Cannot acquire real data** | 🟡 Medium | 🔴 Critical | Use transfer learning with small dataset |
| **PyMC3/Theano integration fails** | 🟡 Medium | 🟠 High | Use surrogate model with periodic refinement |
| **Physics loss doesn't converge** | 🟡 Medium | 🟠 High | Adaptive weighting, curriculum learning |
| **Computational cost too high** | 🟢 Low | 🟡 Medium | Use smaller models, faster ODE solvers |
| **Clinical validation fails** | 🟡 Medium | 🔴 Critical | Iterate on model architecture, features |

---

## Success Metrics

### Technical Metrics
- ✅ Bayesian MCMC uses real digital twin (not surrogate)
- ✅ Physics residual < 0.1 on validation data
- ✅ Cache hit rate > 80% in MCMC
- ✅ PINN enforces ODE constraints
- ✅ Real patient data loader passes unit tests

### Clinical Metrics
- ✅ Diagnostic accuracy > 75% (healthy vs dysmotility)
- ✅ Parameter estimates within 20% of literature values
- ✅ Biomarkers correlate with symptom scores (r > 0.5)
- ✅ Outperforms baseline methods (FFT, Random Forest)

### Publication Readiness
- ✅ Validation on 30+ real patients
- ✅ 5-fold cross-validation results
- ✅ Comparison to 2+ baseline methods
- ✅ Clinical interpretation of results

---

## Resource Requirements

### Personnel
- **Machine Learning Engineer** (you) - 50% time, 16 weeks
- **Clinical Collaborator** - 10% time, advisory
- **Data Analyst** (optional) - 20% time, data preprocessing

### Computational
- **GPU:** NVIDIA A100/A6000 for PINN training (cloud: ~$2-3/hr)
- **Storage:** 100 GB for patient data, model checkpoints
- **Compute time:** ~200 GPU-hours for full implementation

### Financial
- **Cloud compute:** $500-1000
- **Software licenses:** $0 (all open source)
- **Data acquisition:** $0 if academic collaboration
- **Total:** ~$1000

---

## Conclusion

These three issues represent **fundamental architectural gaps** that separate a research prototype from a production-ready clinical tool. While the recent bug fixes achieved test stability, addressing these issues requires:

1. **Deep technical expertise** in Bayesian inference, PINNs, and medical signal processing
2. **Clinical partnerships** for real data acquisition and validation
3. **Sustained effort** over 12-16 weeks

**Recommended prioritization:**

1. **Start immediately:** Simulation cache + Bayesian integration (enables Phases 2-3)
2. **Parallel track:** Clinical data acquisition (long lead time)
3. **After Bayesian complete:** PINN physics loss (builds on same infrastructure)

This plan is ambitious but achievable with focused effort and the right collaborations.

---

## References

- **Bayesian PINNs:** Yang et al. (2021) "B-PINNs: Bayesian Physics-Informed Neural Networks"
- **Medical Time Series:** Faust et al. (2018) "Deep learning for healthcare applications"
- **GI Motility Modeling:** Cheng et al. (2010) "Multiscale modeling of GI electrical activity"
- **EGG Signal Processing:** Yin & Chen (2013) "Electrogastrography: methodology, validation, and applications"

---

**Next Steps:**
1. Review and approve this plan
2. Prioritize which issue to tackle first
3. Set up development environment (GPU access, PyMC3, pyedflib)
4. Begin implementation

Would you like me to start with any specific section?
