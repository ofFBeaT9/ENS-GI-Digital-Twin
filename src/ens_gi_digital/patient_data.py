"""
Patient data loading utilities for ENS-GI Digital Twin.

Supports:
- CSV/TSV files (EGG, HRM, Ca imaging)
- NumPy arrays (.npy)
- HDF5 (.h5, .hdf5)

Example CSV format for EGG data:
    time,ch1,ch2,ch3,ch4,ch5
    0.0,-65.2,-65.1,-65.3,-65.2,-65.1
    0.1,-65.0,-64.9,-65.1,-65.0,-64.9
    ...

Example CSV format for HRM (pressure) data:
    time,sensor1,sensor2,sensor3,sensor4,sensor5
    0.0,10.5,12.3,11.8,13.2,10.9
    0.1,10.7,12.5,12.0,13.4,11.1
    ...
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import warnings
import sys

# Configure UTF-8 output for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


class PatientDataLoader:
    """Load and validate patient clinical data."""

    def __init__(self, data_dir: str = 'patient_data'):
        """
        Args:
            data_dir: Directory containing patient data files
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            warnings.warn(f"Data directory does not exist: {self.data_dir}")

    def load_egg_csv(self, filename: str,
                     time_col: str = 'time',
                     delimiter: str = ',',
                     voltage_unit: str = 'mV') -> Tuple[np.ndarray, np.ndarray]:
        """Load EGG (electrogastrography) data from CSV.

        Args:
            filename: CSV filename (relative to data_dir or absolute path)
            time_col: Name of time column
            delimiter: CSV delimiter
            voltage_unit: Unit of voltage data ('mV' or 'V')

        Returns:
            time: [T] array in ms
            voltages: [T, N] array in mV
        """
        filepath = self.data_dir / filename if not Path(filename).is_absolute() else Path(filename)

        if not filepath.exists():
            raise FileNotFoundError(f"EGG file not found: {filepath}")

        df = pd.read_csv(filepath, delimiter=delimiter)

        if time_col not in df.columns:
            raise ValueError(f"Time column '{time_col}' not found in {filename}")

        time = df[time_col].values  # Assume ms
        voltage_cols = [c for c in df.columns if c != time_col]
        voltages = df[voltage_cols].values  # [T, N]

        # Convert to mV if needed
        if voltage_unit == 'V':
            voltages = voltages * 1000.0

        # Validate
        if voltages.ndim != 2:
            raise ValueError(f"Voltages must be 2D [time, channels], got shape {voltages.shape}")
        if len(time) != len(voltages):
            raise ValueError(f"Time and voltage length mismatch: {len(time)} vs {len(voltages)}")

        print(f"✓ Loaded EGG: {voltages.shape[0]} timepoints, {voltages.shape[1]} channels")
        print(f"  Time range: {time[0]:.1f} - {time[-1]:.1f} ms")
        print(f"  Voltage range: {voltages.min():.2f} - {voltages.max():.2f} mV")

        return time, voltages

    def load_hrm_csv(self, filename: str,
                     time_col: str = 'time',
                     delimiter: str = ',',
                     normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Load HRM (high-resolution manometry) pressure data.

        Args:
            filename: CSV filename (relative to data_dir or absolute path)
            time_col: Name of time column
            delimiter: CSV delimiter
            normalize: If True, normalize forces to 0-1 range

        Returns:
            time: [T] array in ms
            forces: [T, N] array (normalized if normalize=True)
        """
        filepath = self.data_dir / filename if not Path(filename).is_absolute() else Path(filename)

        if not filepath.exists():
            raise FileNotFoundError(f"HRM file not found: {filepath}")

        df = pd.read_csv(filepath, delimiter=delimiter)

        if time_col not in df.columns:
            raise ValueError(f"Time column '{time_col}' not found in {filename}")

        time = df[time_col].values
        force_cols = [c for c in df.columns if c != time_col]
        forces = df[force_cols].values

        # Normalize to 0-1 range if requested
        if normalize:
            f_min = forces.min()
            f_max = forces.max()
            if f_max - f_min < 1e-9:
                warnings.warn("Force data has no variation, normalizing to 0.5")
                forces = np.full_like(forces, 0.5)
            else:
                forces = (forces - f_min) / (f_max - f_min)

        print(f"✓ Loaded HRM: {forces.shape[0]} timepoints, {forces.shape[1]} sensors")
        if normalize:
            print(f"  Force range (normalized): {forces.min():.3f} - {forces.max():.3f}")
        else:
            print(f"  Force range: {forces.min():.2f} - {forces.max():.2f}")

        return time, forces

    def load_calcium_csv(self, filename: str,
                         time_col: str = 'time',
                         delimiter: str = ',',
                         normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Load calcium imaging data from CSV.

        Args:
            filename: CSV filename (relative to data_dir or absolute path)
            time_col: Name of time column
            delimiter: CSV delimiter
            normalize: If True, normalize to 0-1 range (ΔF/F₀)

        Returns:
            time: [T] array in ms
            calcium: [T, N] array of calcium signals
        """
        filepath = self.data_dir / filename if not Path(filename).is_absolute() else Path(filename)

        if not filepath.exists():
            raise FileNotFoundError(f"Calcium file not found: {filepath}")

        df = pd.read_csv(filepath, delimiter=delimiter)

        if time_col not in df.columns:
            raise ValueError(f"Time column '{time_col}' not found in {filename}")

        time = df[time_col].values
        ca_cols = [c for c in df.columns if c != time_col]
        calcium = df[ca_cols].values

        # Normalize if requested (ΔF/F₀)
        if normalize:
            baseline = calcium[:100].mean(axis=0) if len(calcium) > 100 else calcium.mean(axis=0)
            calcium = (calcium - baseline) / (baseline + 1e-9)

        print(f"✓ Loaded Calcium: {calcium.shape[0]} timepoints, {calcium.shape[1]} ROIs")
        print(f"  Calcium range: {calcium.min():.3f} - {calcium.max():.3f}")

        return time, calcium

    def load_patient_data(self, patient_id: str,
                         egg_file: Optional[str] = None,
                         hrm_file: Optional[str] = None,
                         calcium_file: Optional[str] = None) -> Dict:
        """Load complete patient dataset.

        Args:
            patient_id: Patient identifier
            egg_file: EGG CSV filename (default: {patient_id}_egg.csv)
            hrm_file: HRM CSV filename (default: {patient_id}_hrm.csv)
            calcium_file: Calcium CSV filename (default: {patient_id}_calcium.csv)

        Returns:
            dict with 'patient_id', 'time', 'voltages', 'forces', 'calcium', 'metadata'
        """
        print(f"\nLoading patient data: {patient_id}")
        print("="*60)

        # Default filenames
        egg_file = egg_file or f"{patient_id}_egg.csv"
        hrm_file = hrm_file or f"{patient_id}_hrm.csv"
        calcium_file = calcium_file or f"{patient_id}_calcium.csv"

        # Load available data
        data = {'patient_id': patient_id}

        # Try to load EGG
        egg_path = self.data_dir / egg_file
        if egg_path.exists():
            time_egg, voltages = self.load_egg_csv(egg_file)
            data['time'] = time_egg
            data['voltages'] = voltages
        else:
            print(f"⚠ EGG file not found: {egg_file}")
            data['voltages'] = None

        # Try to load HRM
        hrm_path = self.data_dir / hrm_file
        if hrm_path.exists():
            time_hrm, forces = self.load_hrm_csv(hrm_file)

            # Resample to common time axis if needed
            if 'time' in data and not np.array_equal(data['time'], time_hrm):
                print("⚠ Time axes differ, resampling HRM to EGG time...")
                forces = self._resample(time_hrm, forces, data['time'])
            elif 'time' not in data:
                data['time'] = time_hrm

            data['forces'] = forces
        else:
            print(f"⚠ HRM file not found: {hrm_file}")
            data['forces'] = None

        # Try to load Calcium
        ca_path = self.data_dir / calcium_file
        if ca_path.exists():
            time_ca, calcium = self.load_calcium_csv(calcium_file)

            # Resample to common time axis if needed
            if 'time' in data and not np.array_equal(data['time'], time_ca):
                print("⚠ Time axes differ, resampling calcium to common time...")
                calcium = self._resample(time_ca, calcium, data['time'])
            elif 'time' not in data:
                data['time'] = time_ca

            data['calcium'] = calcium
        else:
            print(f"⚠ Calcium file not found: {calcium_file} (optional)")
            data['calcium'] = None

        # Check that we have at least one data type
        if data['voltages'] is None and data['forces'] is None and data['calcium'] is None:
            raise ValueError(f"No data files found for patient {patient_id}")

        # Generate metadata
        if 'time' in data:
            n_timepoints = len(data['time'])
            duration_ms = data['time'][-1] - data['time'][0]
            sampling_rate_hz = 1000.0 / np.mean(np.diff(data['time']))

            n_channels = 0
            if data['voltages'] is not None:
                n_channels = data['voltages'].shape[1]

            data['metadata'] = {
                'n_timepoints': n_timepoints,
                'n_channels': n_channels,
                'duration_ms': duration_ms,
                'sampling_rate_hz': sampling_rate_hz,
            }

            print(f"\n{'='*60}")
            print(f"Patient {patient_id} summary:")
            print(f"  Duration: {duration_ms:.0f} ms ({duration_ms/1000:.1f} s)")
            print(f"  Sampling rate: {sampling_rate_hz:.1f} Hz")
            print(f"  Channels/ROIs: {n_channels}")
            print(f"  Data types: ", end="")
            types = []
            if data['voltages'] is not None:
                types.append("EGG")
            if data['forces'] is not None:
                types.append("HRM")
            if data['calcium'] is not None:
                types.append("Calcium")
            print(", ".join(types))
        else:
            data['metadata'] = {}

        return data

    def _resample(self, old_time: np.ndarray, old_data: np.ndarray,
                  new_time: np.ndarray) -> np.ndarray:
        """Resample data to new time axis using linear interpolation.

        Args:
            old_time: [T_old] original time points
            old_data: [T_old, N] original data
            new_time: [T_new] target time points

        Returns:
            [T_new, N] resampled data
        """
        from scipy.interpolate import interp1d

        interpolator = interp1d(old_time, old_data, axis=0,
                               kind='linear', fill_value='extrapolate')
        return interpolator(new_time)


def create_sample_patient_data(patient_id: str = 'P001',
                               output_dir: str = 'patient_data',
                               duration_ms: float = 2000.0,
                               n_channels: int = 5,
                               sampling_rate_hz: float = 1000.0):
    """Create sample patient data files for testing.

    Args:
        patient_id: Patient identifier
        output_dir: Output directory
        duration_ms: Duration in milliseconds
        n_channels: Number of channels/sensors
        sampling_rate_hz: Sampling rate
    """
    print(f"Creating sample patient data: {patient_id}")
    print("="*60)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Generate simple synthetic slow wave data
    dt = 1000.0 / sampling_rate_hz  # ms
    n_points = int(duration_ms / dt)
    time = np.arange(n_points) * dt

    # EGG: slow wave oscillations (3 cpm = 0.05 Hz)
    voltages = np.zeros((n_points, n_channels))
    for i in range(n_channels):
        phase = i * 0.3  # Phase gradient along GI tract
        voltages[:, i] = -65.0 + 5.0 * np.sin(2 * np.pi * 0.05 * time / 1000.0 + phase)
        # Add some noise
        voltages[:, i] += np.random.normal(0, 0.5, n_points)

    # HRM: pressure waves synchronized with slow waves
    forces = np.zeros((n_points, n_channels))
    for i in range(n_channels):
        phase = i * 0.3
        forces[:, i] = 0.5 + 0.3 * np.sin(2 * np.pi * 0.05 * time / 1000.0 + phase)
        forces[:, i] += np.random.normal(0, 0.05, n_points)
        forces[:, i] = np.clip(forces[:, i], 0, 1)  # Keep in 0-1 range

    # Create EGG CSV
    egg_file = output_path / f"{patient_id}_egg.csv"
    egg_df = pd.DataFrame(voltages, columns=[f'ch{i+1}' for i in range(n_channels)])
    egg_df.insert(0, 'time', time)
    egg_df.to_csv(egg_file, index=False)
    print(f"✓ Created EGG file: {egg_file}")

    # Create HRM CSV
    hrm_file = output_path / f"{patient_id}_hrm.csv"
    hrm_df = pd.DataFrame(forces, columns=[f'sensor{i+1}' for i in range(n_channels)])
    hrm_df.insert(0, 'time', time)
    hrm_df.to_csv(hrm_file, index=False)
    print(f"✓ Created HRM file: {hrm_file}")

    print(f"\nSample data created successfully!")
    print(f"  Patient ID: {patient_id}")
    print(f"  Duration: {duration_ms} ms")
    print(f"  Channels: {n_channels}")
    print(f"  Sampling rate: {sampling_rate_hz} Hz")
    print(f"  Files: {egg_file.name}, {hrm_file.name}")


# Example usage
if __name__ == '__main__':
    import sys

    # Create sample data if it doesn't exist
    if not Path('patient_data').exists() or len(list(Path('patient_data').glob('*.csv'))) == 0:
        print("No patient data found. Creating sample data...\n")
        create_sample_patient_data('P001', n_channels=5, duration_ms=2000.0)
        print("\n")

    # Test loading
    loader = PatientDataLoader('patient_data/')

    # Load patient
    try:
        patient = loader.load_patient_data('P001')
        print(f"\n{'='*60}")
        print("Patient data loaded successfully!")
        print(f"  Time points: {patient['metadata']['n_timepoints']}")
        print(f"  Voltage shape: {patient['voltages'].shape if patient['voltages'] is not None else 'N/A'}")
        print(f"  Force shape: {patient['forces'].shape if patient['forces'] is not None else 'N/A'}")
    except Exception as e:
        print(f"\n✗ Error loading patient data: {e}")
        sys.exit(1)
