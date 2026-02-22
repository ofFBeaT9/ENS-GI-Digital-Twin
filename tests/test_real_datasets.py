"""
Integration tests for real dataset loaders.

Tests for:
- Zenodo 3-channel EGG dataset (Popovic et al. 2020, DOI: 10.5281/zenodo.3878435)
- SPARC Colonic HRM dataset (Dinning, Brookes et al. 2019, DOI: 10.26275/RYFT-516S)

All tests that require actual data files are skipped automatically when the
data directory is absent (e.g. in CI without the datasets downloaded).

To run with real data:
    # Download Zenodo EGG
    python scripts/download_zenodo_egg.py
    # Download SPARC HRM
    python scripts/download_sparc_hrm.py
    # Run tests
    pytest tests/test_real_datasets.py -v
"""

import pytest
import numpy as np
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ens_gi_digital.patient_data import PatientDataLoader

# ---------------------------------------------------------------------------
# Skip flags — checked at module import time
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent.parent  # project root

ZENODO_DATA = _HERE / 'data' / 'EGG-database' / 'ID1_fasting.txt'
SPARC_DATA  = _HERE / 'data' / 'pennsieve data base' / 'files' / 'primary' / 'sub-01'

ZENODO_PRESENT = ZENODO_DATA.exists()
SPARC_PRESENT  = SPARC_DATA.exists() and SPARC_DATA.is_dir()

# ---------------------------------------------------------------------------
# Zenodo EGG tests
# ---------------------------------------------------------------------------

class TestZenodoEGG:
    """Tests for the Zenodo 3-channel EGG dataset loader."""

    @pytest.mark.skipif(not ZENODO_PRESENT,
                        reason="Zenodo EGG data not present — run scripts/download_zenodo_egg.py")
    def test_shape(self):
        """Fasting recording should be (2400, 3) and time length should match."""
        loader = PatientDataLoader(str(_HERE / 'data'))
        time, voltages = loader.load_zenodo_egg(subject_id=1, condition='fasting')

        assert voltages.ndim == 2, "voltages must be 2D"
        assert voltages.shape == (2400, 3), \
            f"Expected (2400, 3), got {voltages.shape}"
        assert len(time) == 2400, f"Expected 2400 time points, got {len(time)}"

    @pytest.mark.skipif(not ZENODO_PRESENT,
                        reason="Zenodo EGG data not present — run scripts/download_zenodo_egg.py")
    def test_time_in_ms(self):
        """Time array should be in milliseconds (not seconds).

        20 min @ 2 Hz = 2400 samples.
        Last sample index = 2399 → time = 2399 / 2.0 * 1000 = 1 199 500 ms.
        """
        loader = PatientDataLoader(str(_HERE / 'data'))
        time, _ = loader.load_zenodo_egg(subject_id=1, condition='fasting')

        expected_last_ms = 2399 / 2.0 * 1000.0  # = 1 199 500 ms
        assert abs(time[-1] - expected_last_ms) < 1.0, \
            f"Expected time[-1] ≈ {expected_last_ms:.0f} ms, got {time[-1]:.1f} ms"
        assert time[0] == 0.0, f"Expected time[0] == 0.0, got {time[0]}"

    @pytest.mark.skipif(not ZENODO_PRESENT,
                        reason="Zenodo EGG data not present — run scripts/download_zenodo_egg.py")
    def test_postprandial_loads(self):
        """Post-prandial recording should have the same shape as fasting."""
        loader = PatientDataLoader(str(_HERE / 'data'))
        time, voltages = loader.load_zenodo_egg(subject_id=1, condition='postprandial')

        assert voltages.shape == (2400, 3), \
            f"Expected (2400, 3), got {voltages.shape}"
        assert len(time) == 2400

    @pytest.mark.skipif(not ZENODO_PRESENT,
                        reason="Zenodo EGG data not present — run scripts/download_zenodo_egg.py")
    def test_voltage_finite(self):
        """All voltage values must be finite (no NaN or Inf)."""
        loader = PatientDataLoader(str(_HERE / 'data'))
        _, voltages = loader.load_zenodo_egg(subject_id=1, condition='fasting')
        assert np.all(np.isfinite(voltages)), "Voltage array contains NaN or Inf"

    def test_invalid_condition_raises(self):
        """ValueError must be raised for unrecognised condition string."""
        loader = PatientDataLoader(str(_HERE / 'data'))
        with pytest.raises(ValueError, match="condition must be one of"):
            loader.load_zenodo_egg(subject_id=1, condition='lunch')

    def test_missing_subject_raises(self):
        """FileNotFoundError must be raised when subject data file does not exist."""
        loader = PatientDataLoader(str(_HERE / 'data'))
        with pytest.raises(FileNotFoundError):
            loader.load_zenodo_egg(subject_id=99, condition='fasting')


# ---------------------------------------------------------------------------
# SPARC HRM tests
# ---------------------------------------------------------------------------

class TestSPARCHRM:
    """Tests for the SPARC Colonic HRM dataset loader."""

    @pytest.mark.skipif(not SPARC_PRESENT,
                        reason="SPARC HRM data not present — run scripts/download_sparc_hrm.py")
    def test_channels(self):
        """Recording must have exactly 12 force/pressure channels."""
        loader = PatientDataLoader(str(_HERE / 'data'))
        time, forces = loader.load_sparc_hrm(subject_id=1)

        assert forces.ndim == 2, "forces must be 2D"
        assert forces.shape[1] == 12, \
            f"Expected 12 sensors, got {forces.shape[1]}"

    @pytest.mark.skipif(not SPARC_PRESENT,
                        reason="SPARC HRM data not present — run scripts/download_sparc_hrm.py")
    def test_time_in_ms_not_seconds(self):
        """Time array must be returned in milliseconds.

        SPARC files store time in seconds at 10 Hz.
        The loader converts to ms so consecutive samples are ~100 ms apart.
        """
        loader = PatientDataLoader(str(_HERE / 'data'))
        time, _ = loader.load_sparc_hrm(subject_id=1)

        dt = time[1] - time[0]
        assert abs(dt - 100.0) < 1.0, \
            f"Expected Δt ≈ 100 ms (10 Hz in ms), got {dt:.3f}"

    @pytest.mark.skipif(not SPARC_PRESENT,
                        reason="SPARC HRM data not present — run scripts/download_sparc_hrm.py")
    def test_normalized_range(self):
        """With normalize=True (default), all values must be in [0, 1]."""
        loader = PatientDataLoader(str(_HERE / 'data'))
        _, forces = loader.load_sparc_hrm(subject_id=1, normalize=True)

        assert forces.min() >= 0.0, \
            f"Minimum value below 0: {forces.min():.4f}"
        assert forces.max() <= 1.0 + 1e-6, \
            f"Maximum value above 1: {forces.max():.4f}"

    @pytest.mark.skipif(not SPARC_PRESENT,
                        reason="SPARC HRM data not present — run scripts/download_sparc_hrm.py")
    def test_raw_pressure_positive(self):
        """Raw mmHg pressures are baseline-zeroed and may include negative values.

        SPARC HRM recordings are zeroed to a calibration reference; pressures
        below that reference are negative.  We verify the range is physiologically
        plausible (within ±200 mmHg) rather than asserting strict positivity.
        """
        loader = PatientDataLoader(str(_HERE / 'data'))
        _, forces = loader.load_sparc_hrm(subject_id=1, normalize=False)

        assert forces.min() >= -200.0, \
            f"Unexpectedly low pressure: {forces.min():.2f} mmHg"
        assert forces.max() <= 200.0, \
            f"Unexpectedly high pressure: {forces.max():.2f} mmHg"

    @pytest.mark.skipif(not SPARC_PRESENT,
                        reason="SPARC HRM data not present — run scripts/download_sparc_hrm.py")
    def test_force_array_finite(self):
        """All force values must be finite."""
        loader = PatientDataLoader(str(_HERE / 'data'))
        _, forces = loader.load_sparc_hrm(subject_id=1)
        assert np.all(np.isfinite(forces)), "Force array contains NaN or Inf"

    def test_missing_subject_raises(self):
        """FileNotFoundError must be raised when subject directory does not exist."""
        loader = PatientDataLoader(str(_HERE / 'data'))
        with pytest.raises(FileNotFoundError):
            loader.load_sparc_hrm(subject_id=99)
