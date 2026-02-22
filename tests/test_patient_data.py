"""Unit tests for PatientDataLoader and ClinicalDataLoader.

Covers CSV loading (EGG, HRM), headerless whitespace-delimited files,
time-unit conversion, skip_cols, and the EDF loader error paths.
All tests are self-contained — they create synthetic CSV files via
pytest's tmp_path fixture and do not require real patient data on disk.
"""
import numpy as np
import pytest
from pathlib import Path

from ens_gi_digital.patient_data import PatientDataLoader, ClinicalDataLoader


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _write_egg_csv(path: Path, n_points: int = 20, n_channels: int = 3) -> Path:
    """Write a minimal EGG CSV with a time column."""
    time = np.arange(n_points) * 1.0  # 1 ms steps
    voltages = np.random.randn(n_points, n_channels) * 5.0 - 65.0
    with open(path, 'w') as f:
        header = 'time,' + ','.join(f'ch{i+1}' for i in range(n_channels))
        f.write(header + '\n')
        for t, row in zip(time, voltages):
            f.write(f"{t}," + ','.join(f"{v:.4f}" for v in row) + '\n')
    return path


def _write_egg_headerless(path: Path, n_points: int = 10, n_channels: int = 3) -> Path:
    """Write a headerless whitespace-delimited EGG file (Zenodo format)."""
    voltages = np.random.randn(n_points, n_channels) * 5.0 - 65.0
    with open(path, 'w') as f:
        for row in voltages:
            f.write('  '.join(f"{v:8.4f}" for v in row) + '\n')
    return path


def _write_hrm_csv(path: Path, n_points: int = 20, n_sensors: int = 4,
                   time_unit: str = 'ms') -> Path:
    """Write a minimal HRM CSV."""
    if time_unit == 's':
        time = np.arange(n_points) * 0.1  # 0.1 s steps
    else:
        time = np.arange(n_points) * 100.0  # 100 ms steps
    forces = np.abs(np.random.randn(n_points, n_sensors)) * 10.0 + 5.0
    with open(path, 'w') as f:
        header = 'time,' + ','.join(f'sensor{i+1}' for i in range(n_sensors))
        f.write(header + '\n')
        for t, row in zip(time, forces):
            f.write(f"{t}," + ','.join(f"{v:.4f}" for v in row) + '\n')
    return path


def _write_hrm_csv_with_marker(path: Path, n_points: int = 20,
                                n_sensors: int = 4) -> Path:
    """Write HRM CSV that includes an extra 'Marker' column."""
    time = np.arange(n_points) * 100.0
    marker = np.zeros(n_points)
    forces = np.abs(np.random.randn(n_points, n_sensors)) * 10.0 + 5.0
    with open(path, 'w') as f:
        sensor_cols = ','.join(f'sensor{i+1}' for i in range(n_sensors))
        f.write(f'time,Marker,{sensor_cols}\n')
        for i in range(n_points):
            sensor_vals = ','.join(f"{forces[i, j]:.4f}" for j in range(n_sensors))
            f.write(f"{time[i]},{marker[i]},{sensor_vals}\n")
    return path


# ─────────────────────────────────────────────────────────────
# EGG CSV loader
# ─────────────────────────────────────────────────────────────

class TestEggCSVLoader:
    """Tests for PatientDataLoader.load_egg_csv()."""

    def test_load_standard_csv(self, tmp_path):
        """Standard CSV with time column loads correctly."""
        csv_file = _write_egg_csv(tmp_path / 'egg.csv', n_points=50, n_channels=3)
        loader = PatientDataLoader(str(tmp_path))

        time, voltages = loader.load_egg_csv('egg.csv')

        assert len(time) == 50
        assert voltages.shape == (50, 3)
        assert np.isfinite(voltages).all()

    def test_load_headerless_whitespace(self, tmp_path):
        """Headerless whitespace-delimited file (Zenodo format) loads correctly."""
        txt_file = _write_egg_headerless(tmp_path / 'egg.txt', n_points=10, n_channels=3)
        loader = PatientDataLoader(str(tmp_path))

        time, voltages = loader.load_egg_csv(
            'egg.txt',
            delimiter=r'\s+',
            has_time_col=False,
            sampling_rate_hz=2.0,
        )

        assert len(time) == 10
        assert voltages.shape == (10, 3)
        # Time should be in ms; at 2 Hz step = 500 ms
        assert abs(time[1] - time[0] - 500.0) < 1e-6

    def test_missing_file_raises(self, tmp_path):
        """FileNotFoundError is raised when the CSV does not exist."""
        loader = PatientDataLoader(str(tmp_path))
        with pytest.raises(FileNotFoundError):
            loader.load_egg_csv('nonexistent.csv')

    def test_missing_time_col_raises(self, tmp_path):
        """ValueError is raised when the declared time column is absent."""
        # Write a CSV without a 'time' column
        csv_path = tmp_path / 'notimecol.csv'
        csv_path.write_text('ch1,ch2\n1.0,2.0\n3.0,4.0\n')
        loader = PatientDataLoader(str(tmp_path))

        with pytest.raises(ValueError, match="Time column"):
            loader.load_egg_csv('notimecol.csv', time_col='time')


# ─────────────────────────────────────────────────────────────
# HRM CSV loader
# ─────────────────────────────────────────────────────────────

class TestHRMCSVLoader:
    """Tests for PatientDataLoader.load_hrm_csv()."""

    def test_load_hrm_normalized(self, tmp_path):
        """HRM CSV loads with normalized output in [0, 1]."""
        _write_hrm_csv(tmp_path / 'hrm.csv', n_points=30, n_sensors=4)
        loader = PatientDataLoader(str(tmp_path))

        time, forces = loader.load_hrm_csv('hrm.csv', normalize=True)

        assert forces.shape == (30, 4)
        assert forces.min() >= 0.0 - 1e-9
        assert forces.max() <= 1.0 + 1e-9

    def test_load_hrm_raw(self, tmp_path):
        """HRM CSV loads raw (unnormalized) values when normalize=False."""
        _write_hrm_csv(tmp_path / 'hrm.csv', n_points=30, n_sensors=4)
        loader = PatientDataLoader(str(tmp_path))

        time, forces = loader.load_hrm_csv('hrm.csv', normalize=False)

        assert forces.shape == (30, 4)
        # Raw values should be positive (synthetically generated as abs(...)+5)
        assert forces.min() > 0.0

    def test_time_unit_seconds(self, tmp_path):
        """time_unit='s' multiplies time values by 1000 to give ms."""
        _write_hrm_csv(tmp_path / 'hrm_s.csv', n_points=20, n_sensors=3,
                       time_unit='s')
        loader = PatientDataLoader(str(tmp_path))

        time, forces = loader.load_hrm_csv('hrm_s.csv', time_unit='s', normalize=False)

        # Each step was 0.1 s → should be 100 ms after conversion
        step = time[1] - time[0]
        assert abs(step - 100.0) < 1e-6, f"Expected 100 ms step, got {step}"

    def test_skip_cols(self, tmp_path):
        """skip_cols removes named columns from the sensor data."""
        _write_hrm_csv_with_marker(tmp_path / 'hrm_marker.csv',
                                   n_points=20, n_sensors=4)
        loader = PatientDataLoader(str(tmp_path))

        time, forces = loader.load_hrm_csv(
            'hrm_marker.csv',
            skip_cols=['Marker'],
            normalize=False,
        )

        # Only 4 sensor columns remain (Marker column was dropped)
        assert forces.shape == (20, 4)


# ─────────────────────────────────────────────────────────────
# EDF loader (error paths only — no pyedflib required)
# ─────────────────────────────────────────────────────────────

class TestEDFLoader:
    """Tests for PatientDataLoader.load_edf() — error paths."""

    def test_edf_file_not_found(self, tmp_path):
        """FileNotFoundError (or ImportError if pyedflib absent) when EDF missing."""
        loader = PatientDataLoader(str(tmp_path))
        with pytest.raises((FileNotFoundError, ImportError)):
            loader.load_edf('nonexistent.edf')

    def test_edf_import_error_without_pyedflib(self, tmp_path, monkeypatch):
        """ImportError is raised when pyedflib is not installed."""
        import sys

        # Patch pyedflib out of the import system
        monkeypatch.setitem(sys.modules, 'pyedflib', None)

        loader = PatientDataLoader(str(tmp_path))
        # Create a dummy file so the existence check passes
        (tmp_path / 'test.edf').touch()

        with pytest.raises((ImportError, TypeError)):
            loader.load_edf('test.edf')
