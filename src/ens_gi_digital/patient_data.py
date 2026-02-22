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
                     voltage_unit: str = 'mV',
                     has_time_col: bool = True,
                     sampling_rate_hz: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Load EGG (electrogastrography) data from CSV or plain-text file.

        Supports two formats:
        - Standard CSV with a time column (has_time_col=True, default)
        - Headerless plain-text files with no time column, e.g. Zenodo EGG
          dataset (has_time_col=False, requires sampling_rate_hz)

        Args:
            filename: Filename relative to data_dir, or absolute path.
            time_col: Name of the time column (used when has_time_col=True).
            delimiter: Column delimiter (',', '\\t', ' ', etc.).
            voltage_unit: Unit of voltage data — 'mV' (default) or 'V'.
            has_time_col: If False, the file has no time column and no header
                row (e.g. Zenodo three-channel EGG .txt files). A time array
                is generated from sampling_rate_hz.
            sampling_rate_hz: Sampling rate in Hz. Required when
                has_time_col=False; ignored otherwise.

        Returns:
            time:     [T] array in ms
            voltages: [T, N] array in mV
        """
        filepath = self.data_dir / filename if not Path(filename).is_absolute() else Path(filename)

        if not filepath.exists():
            raise FileNotFoundError(f"EGG file not found: {filepath}")

        if not has_time_col:
            # Headerless file — all columns are signal channels
            if sampling_rate_hz is None:
                raise ValueError(
                    "sampling_rate_hz is required when has_time_col=False"
                )
            _engine = 'python' if len(delimiter) > 1 else 'c'
            df = pd.read_csv(filepath, sep=delimiter, header=None, engine=_engine)
            df.columns = [f'ch{i+1}' for i in range(df.shape[1])]
            voltages = df.values.astype(np.float64)
            time = np.arange(len(voltages)) / sampling_rate_hz * 1000.0  # s → ms
        else:
            _engine = 'python' if len(delimiter) > 1 else 'c'
            df = pd.read_csv(filepath, sep=delimiter, engine=_engine)
            if time_col not in df.columns:
                raise ValueError(f"Time column '{time_col}' not found in {filename}")
            time = df[time_col].values  # assumed ms
            voltage_cols = [c for c in df.columns if c != time_col]
            voltages = df[voltage_cols].values.astype(np.float64)

        # Convert to mV if needed
        if voltage_unit == 'V':
            voltages = voltages * 1000.0

        # Validate
        if voltages.ndim != 2:
            raise ValueError(f"Voltages must be 2D [time, channels], got shape {voltages.shape}")
        if len(time) != len(voltages):
            raise ValueError(f"Time and voltage length mismatch: {len(time)} vs {len(voltages)}")

        print(f"[OK] Loaded EGG: {voltages.shape[0]} timepoints, {voltages.shape[1]} channels")
        print(f"  Time range: {time[0]:.1f} - {time[-1]:.1f} ms")
        print(f"  Voltage range: {voltages.min():.2f} - {voltages.max():.2f} mV")

        return time, voltages

    def load_hrm_csv(self, filename: str,
                     time_col: str = 'time',
                     delimiter: str = ',',
                     normalize: bool = True,
                     time_unit: str = 'ms',
                     skip_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Load HRM (high-resolution manometry) pressure data.

        Supports standard CSV files and the SPARC colonic HRM format
        (tab-delimited, time in seconds, extra marker column).

        Args:
            filename: Filename relative to data_dir, or absolute path.
            time_col: Name of the time column.
            delimiter: Column delimiter (',', '\\t', ' ', etc.).
            normalize: If True, normalise forces to 0-1 range.
            time_unit: Unit of the time column — 'ms' (default) or 's'.
                       If 's', the time values are multiplied by 1000 so the
                       returned array is always in milliseconds.
            skip_cols: List of column names to drop before extracting force
                       channels.  Use this to discard e.g. the 'Marker channel'
                       column present in SPARC HRM files.

        Returns:
            time:   [T] array in ms
            forces: [T, N] array (normalised to 0-1 if normalize=True,
                    otherwise raw pressure values in original units)
        """
        filepath = self.data_dir / filename if not Path(filename).is_absolute() else Path(filename)

        if not filepath.exists():
            raise FileNotFoundError(f"HRM file not found: {filepath}")

        df = pd.read_csv(filepath, delimiter=delimiter)

        if time_col not in df.columns:
            raise ValueError(f"Time column '{time_col}' not found in {filename}")

        time = df[time_col].values
        if time_unit == 's':
            time = time * 1000.0  # s → ms

        # Drop unwanted columns (e.g. SPARC marker channel)
        drop = set(skip_cols or []) | {time_col}
        force_cols = [c for c in df.columns if c not in drop]
        forces = df[force_cols].values.astype(np.float64)

        # Normalise to 0-1 range if requested
        if normalize:
            f_min = forces.min()
            f_max = forces.max()
            if f_max - f_min < 1e-9:
                warnings.warn("Force data has no variation, normalizing to 0.5")
                forces = np.full_like(forces, 0.5)
            else:
                forces = (forces - f_min) / (f_max - f_min)

        print(f"[OK] Loaded HRM: {forces.shape[0]} timepoints, {forces.shape[1]} sensors")
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

        print(f"[OK] Loaded Calcium: {calcium.shape[0]} timepoints, {calcium.shape[1]} ROIs")
        print(f"  Calcium range: {calcium.min():.3f} - {calcium.max():.3f}")

        return time, calcium

    def load_edf(self, filename: str,
                 channel_indices: Optional[List[int]] = None,
                 voltage_unit: str = 'mV') -> Tuple[np.ndarray, np.ndarray, float, List[str]]:
        """Load EEG/EGG data from an EDF (European Data Format) file.

        EDF is the clinical standard for storing electrophysiology recordings
        (EGG, EEG, ECG).  Requires ``pyedflib`` (``pip install pyedflib``).

        Args:
            filename:        EDF filename relative to data_dir, or absolute path.
            channel_indices: List of channel indices to load (None = all channels).
            voltage_unit:    Physical unit of the stored signal ('mV' or 'V').
                             If 'V', signals are multiplied by 1000 to convert to mV.

        Returns:
            time         - [T] time array in ms
            voltages     - [T, n_channels] signal array in mV
            sampling_rate - sampling frequency in Hz
            channel_names - list of channel label strings

        Raises:
            ImportError  if pyedflib is not installed.
            FileNotFoundError if the EDF file does not exist.
        """
        try:
            import pyedflib
        except ImportError as exc:
            raise ImportError(
                "pyedflib is required to load EDF files. "
                "Install with: pip install pyedflib"
            ) from exc

        filepath = (self.data_dir / filename
                    if not Path(filename).is_absolute() else Path(filename))

        if not filepath.exists():
            raise FileNotFoundError(f"EDF file not found: {filepath}")

        f = pyedflib.EdfReader(str(filepath))
        n_channels_total = f.signals_in_file
        all_labels = [f.getLabel(i) for i in range(n_channels_total)]

        if channel_indices is None:
            channel_indices = list(range(n_channels_total))

        sampling_rate = float(f.getSampleFrequency(channel_indices[0]))

        signals = []
        for idx in channel_indices:
            signals.append(f.readSignal(idx))
        f.close()

        channels = np.array(signals).T.astype(np.float64)  # [T, n_channels]

        if voltage_unit == 'V':
            channels = channels * 1000.0  # V → mV

        # Build time array in milliseconds
        n_samples = channels.shape[0]
        time_ms = np.arange(n_samples) / sampling_rate * 1000.0

        channel_names = [all_labels[i] for i in channel_indices]

        print(f"[OK] Loaded EDF: {channels.shape[0]} timepoints, "
              f"{channels.shape[1]} channels @ {sampling_rate} Hz")
        print(f"  Duration: {time_ms[-1] / 1000:.1f} s  |  "
              f"Channels: {channel_names}")
        print(f"  Signal range: {channels.min():.3f} – {channels.max():.3f} mV")

        return time_ms, channels, sampling_rate, channel_names

    # ------------------------------------------------------------------
    # Dataset-specific convenience wrappers
    # ------------------------------------------------------------------

    def load_zenodo_egg(self, subject_id: int,
                        condition: str = 'fasting') -> Tuple[np.ndarray, np.ndarray]:
        """Load one recording from the Zenodo 3-channel EGG dataset.

        Reference: Popovic et al. (2020). DOI: 10.5281/zenodo.3878435
        Expected directory layout::

            <data_dir>/zenodo_egg/ID1_fasting.txt
            <data_dir>/zenodo_egg/ID1_postprandial.txt
            ...

        Files are plain-text, 3 columns (CH1 CH2 CH3), space-delimited,
        no header row, sampled at 2 Hz.

        Args:
            subject_id: Integer subject number (1–20).
            condition:  'fasting' or 'postprandial'.

        Returns:
            time:     [2400] array in ms (0 – 1 199 500 ms ≈ 20 min)
            voltages: [2400, 3] array in mV
        """
        valid = ('fasting', 'postprandial')
        if condition not in valid:
            raise ValueError(f"condition must be one of {valid}, got '{condition}'")

        filename = f"EGG-database/ID{subject_id}_{condition}.txt"
        return self.load_egg_csv(
            filename,
            delimiter=r'\s+',
            has_time_col=False,
            sampling_rate_hz=2.0,
        )

    def load_sparc_hrm(self, subject_id: int,
                       normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Load one recording from the SPARC Colonic HRM dataset.

        Reference: Dinning, Brookes et al. (2019). DOI: 10.26275/RYFT-516S
        Expected directory layout::

            <data_dir>/sparc_hrm/sub-01/01_F_31071947.txt
            <data_dir>/sparc_hrm/sub-02/...

        Files are tab-delimited with 14 columns:
        ``time (s)  |  Marker channel  |  sensor 1  ...  sensor 12``

        Sampling rate: 10 Hz.  Pressure units: mmHg.

        Args:
            subject_id: SPARC subject number (1–63; note some IDs are missing).
            normalize:  If True, normalise pressure to 0-1 range.

        Returns:
            time:   [T] array in ms
            forces: [T, 12] array (mmHg, or normalised if normalize=True)
        """
        # Build the subject directory name (zero-padded to 2 digits)
        sub_dir = f"sub-{subject_id:02d}"

        # Actual layout: <data_dir>/pennsieve data base/files/primary/sub-NN/
        base = self.data_dir / 'pennsieve data base' / 'files' / 'primary' / sub_dir
        if not base.exists():
            raise FileNotFoundError(
                f"SPARC HRM subject directory not found: {base}"
            )

        # Navigate into sam-* subdirectory if present (one extra nesting level
        # in the Pennsieve download layout: sub-01/sam-01-F-31071947/)
        sam_dirs = [d for d in base.iterdir() if d.is_dir() and d.name.startswith('sam-')]
        scan_dir = sam_dirs[0] if sam_dirs else base

        # Find the primary .txt data file (skip _params / _header files)
        candidates = [
            f for f in scan_dir.iterdir()
            if f.suffix == '.txt'
            and not f.stem.endswith('_params')
            and not f.stem.endswith('_header')
        ]
        if not candidates:
            raise FileNotFoundError(
                f"No primary HRM .txt file found in {scan_dir}"
            )
        data_file = candidates[0]

        # Read headerless tab-delimited file; engine='python' handles \r-only line endings
        # Layout: col 0 = time (s)  |  col 1 = marker channel (skip)  |  cols 2-13 = 12 sensors
        df = pd.read_csv(str(data_file), sep='\t', header=None, engine='python')

        time = df.iloc[:, 0].values.astype(np.float64) * 1000.0  # s → ms
        forces = df.iloc[:, 2:14].values.astype(np.float64)

        if normalize:
            f_min, f_max = forces.min(), forces.max()
            if f_max - f_min < 1e-9:
                warnings.warn("Force data has no variation, normalizing to 0.5")
                forces = np.full_like(forces, 0.5)
            else:
                forces = (forces - f_min) / (f_max - f_min)

        print(f"[OK] Loaded SPARC HRM: {forces.shape[0]} timepoints, {forces.shape[1]} sensors")
        if normalize:
            print(f"  Force range (normalized): {forces.min():.3f} - {forces.max():.3f}")
        else:
            print(f"  Force range: {forces.min():.2f} - {forces.max():.2f}")

        return time, forces

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
    print(f"[OK] Created EGG file: {egg_file}")

    # Create HRM CSV
    hrm_file = output_path / f"{patient_id}_hrm.csv"
    hrm_df = pd.DataFrame(forces, columns=[f'sensor{i+1}' for i in range(n_channels)])
    hrm_df.insert(0, 'time', time)
    hrm_df.to_csv(hrm_file, index=False)
    print(f"[OK] Created HRM file: {hrm_file}")

    print(f"\nSample data created successfully!")
    print(f"  Patient ID: {patient_id}")
    print(f"  Duration: {duration_ms} ms")
    print(f"  Channels: {n_channels}")
    print(f"  Sampling rate: {sampling_rate_hz} Hz")
    print(f"  Files: {egg_file.name}, {hrm_file.name}")


# ═══════════════════════════════════════════════════════════════
# Clinical Data Loader — supports CSV and EDF (medical standard)
# ═══════════════════════════════════════════════════════════════

class ClinicalDataLoader:
    """
    Load and preprocess real patient data (EGG, HRM, metadata).

    Supports multiple formats:
    - CSV (simple tabular)
    - EDF (European Data Format - medical standard via pyedflib)
    - JSON (metadata)

    Usage:
        loader = ClinicalDataLoader('path/to/data', dataset_type='egg')
        raw = loader.load_patient('P001')
        preprocessed = loader.preprocess_egg(raw['egg'])
    """

    def __init__(self, data_root: str, dataset_type: str = 'egg'):
        """
        Args:
            data_root: Root directory containing patient data
            dataset_type: 'egg', 'hrm', or 'multimodal'
        """
        self.data_root = Path(data_root)
        self.dataset_type = dataset_type
        self.metadata_cache: Dict = {}

    def load_patient(self, patient_id: str) -> Dict:
        """
        Load all available data for a patient.

        Returns:
            {
                'patient_id': str,
                'egg': {
                    'time': np.array,
                    'channels': np.array,   # [T, N]
                    'sampling_rate': float,
                    'channel_names': list,
                },
                'hrm': {...},
                'metadata': {...},
            }
        """
        result: Dict = {'patient_id': patient_id}

        # EGG data
        edf_path = self.data_root / f"{patient_id}.edf"
        egg_csv_path = self.data_root / f"{patient_id}_egg.csv"

        if edf_path.exists():
            result['egg'] = self._load_edf(patient_id)
        elif egg_csv_path.exists():
            result['egg'] = self._load_egg_csv(patient_id)
        else:
            result['egg'] = None
            warnings.warn(f"No EGG data found for patient {patient_id}")

        # HRM data
        hrm_path = self.data_root / f"{patient_id}_hrm.csv"
        if hrm_path.exists():
            result['hrm'] = self._load_hrm_csv(patient_id)
        else:
            result['hrm'] = None

        # Metadata
        result['metadata'] = self._load_metadata(patient_id)

        return result

    # ── EDF loader ─────────────────────────────────────────────

    def _load_edf(self, patient_id: str) -> Dict:
        """Load EDF (European Data Format) file using pyedflib."""
        try:
            import pyedflib
        except ImportError:
            raise ImportError(
                "pyedflib is required for EDF support. "
                "Install with: pip install pyedflib"
            )

        edf_path = self.data_root / f"{patient_id}.edf"
        f = pyedflib.EdfReader(str(edf_path))

        try:
            n_channels = f.signals_in_file
            channel_names = [f.getLabel(i).strip() for i in range(n_channels)]
            sampling_rates = [f.getSampleFrequency(i) for i in range(n_channels)]
            sampling_rate = float(sampling_rates[0]) if sampling_rates else 1.0

            signals = []
            for i in range(n_channels):
                signals.append(f.readSignal(i))
        finally:
            f.close()

        # Align to shortest channel (handles minor length differences)
        min_len = min(len(s) for s in signals)
        channels = np.column_stack([s[:min_len] for s in signals])  # [T, N]
        time = np.arange(min_len) / sampling_rate  # seconds

        print(f"[OK] Loaded EDF {patient_id}: {channels.shape}, fs={sampling_rate} Hz")
        return {
            'time': time,
            'channels': channels,
            'sampling_rate': sampling_rate,
            'channel_names': channel_names,
            'source': 'edf',
        }

    # ── CSV loaders ─────────────────────────────────────────────

    def _load_egg_csv(self, patient_id: str) -> Dict:
        """Load EGG data from CSV file."""
        path = self.data_root / f"{patient_id}_egg.csv"
        df = pd.read_csv(path)
        time = df['time'].values if 'time' in df.columns else np.arange(len(df))
        voltage_cols = [c for c in df.columns if c != 'time']
        channels = df[voltage_cols].values
        sampling_rate = 1000.0 / float(np.mean(np.diff(time))) if len(time) > 1 else 1.0

        print(f"[OK] Loaded EGG CSV {patient_id}: {channels.shape}")
        return {
            'time': time,
            'channels': channels,
            'sampling_rate': sampling_rate,
            'channel_names': voltage_cols,
            'source': 'csv',
        }

    def _load_hrm_csv(self, patient_id: str) -> Dict:
        """Load HRM data from CSV file."""
        path = self.data_root / f"{patient_id}_hrm.csv"
        df = pd.read_csv(path)
        time = df['time'].values if 'time' in df.columns else np.arange(len(df))
        sensor_cols = [c for c in df.columns if c != 'time']
        channels = df[sensor_cols].values
        sampling_rate = 1000.0 / float(np.mean(np.diff(time))) if len(time) > 1 else 1.0

        print(f"[OK] Loaded HRM CSV {patient_id}: {channels.shape}")
        return {
            'time': time,
            'channels': channels,
            'sampling_rate': sampling_rate,
            'channel_names': sensor_cols,
            'source': 'csv',
        }

    # ── Metadata ────────────────────────────────────────────────

    def _load_metadata(self, patient_id: str) -> Dict:
        """Load patient metadata from JSON file (if available)."""
        if patient_id in self.metadata_cache:
            return self.metadata_cache[patient_id]

        json_path = self.data_root / f"{patient_id}_metadata.json"
        if json_path.exists():
            import json as _json
            with open(json_path, 'r') as f:
                metadata = _json.load(f)
        else:
            metadata = {
                'age': None,
                'sex': None,
                'diagnosis': None,
                'medications': [],
                'symptom_scores': {},
            }

        self.metadata_cache[patient_id] = metadata
        return metadata

    # ── Preprocessing ───────────────────────────────────────────

    def preprocess_egg(self, egg_data: Dict,
                       bandpass: Tuple[float, float] = (0.015, 0.15),
                       artifact_threshold_sigma: float = 5.0,
                       normalize: bool = True) -> Dict:
        """
        Preprocess EGG signal: detrend → bandpass filter → artifact rejection → normalize.

        Args:
            egg_data: Raw EGG dict from load_patient() (must have 'channels' and 'sampling_rate')
            bandpass: Frequency band in Hz (default: 0.015–0.15 Hz = gastric slow waves)
            artifact_threshold_sigma: Z-score threshold for artifact rejection
            normalize: If True, z-score normalize each channel

        Returns:
            Copy of egg_data with 'channels' replaced by preprocessed signals,
            plus a 'preprocessing' dict describing steps applied.
        """
        from scipy.signal import butter, filtfilt, detrend as sp_detrend

        channels = egg_data['channels'].copy().astype(float)  # [T, N]
        fs = float(egg_data['sampling_rate'])

        # 1. Detrend (remove baseline drift)
        channels = sp_detrend(channels, axis=0)

        # 2. Bandpass filter (isolate gastric slow waves)
        nyq = fs / 2.0
        low, high = bandpass
        # Guard: if nyq is too low, skip filter
        if high < nyq and low > 0:
            b, a = butter(4, [low / nyq, high / nyq], btype='band')
            channels = filtfilt(b, a, channels, axis=0)
        else:
            warnings.warn(
                f"Bandpass ({low}–{high} Hz) incompatible with sampling rate ({fs} Hz). "
                "Skipping bandpass filter."
            )

        # 3. Artifact rejection — interpolate over high-amplitude samples
        threshold = artifact_threshold_sigma * np.std(channels)
        artifact_mask = np.any(np.abs(channels) > threshold, axis=1)
        n_artifacts = int(np.sum(artifact_mask))

        if n_artifacts > 0 and n_artifacts < len(channels) - 2:
            good_idx = np.where(~artifact_mask)[0]
            bad_idx = np.where(artifact_mask)[0]
            for ch in range(channels.shape[1]):
                channels[bad_idx, ch] = np.interp(bad_idx, good_idx, channels[good_idx, ch])

        # 4. Z-score normalization per channel
        if normalize:
            means = channels.mean(axis=0)
            stds = channels.std(axis=0)
            stds[stds < 1e-10] = 1.0  # avoid divide-by-zero
            channels = (channels - means) / stds

        preprocessed = dict(egg_data)
        preprocessed['channels'] = channels
        preprocessed['preprocessing'] = {
            'detrend': True,
            'bandpass_hz': bandpass,
            'bandpass_applied': high < nyq and low > 0,
            'artifact_threshold_sigma': artifact_threshold_sigma,
            'n_artifacts_removed': n_artifacts,
            'normalization': 'zscore' if normalize else None,
        }

        print(f"[OK] Preprocessed EGG: {n_artifacts} artifacts removed, "
              f"bandpass {low}–{high} Hz, normalized={normalize}")
        return preprocessed

    def preprocess_hrm(self, hrm_data: Dict,
                       smooth_sigma_ms: float = 100.0,
                       normalize: bool = True) -> Dict:
        """
        Preprocess HRM pressure data: smooth → normalize.

        Args:
            hrm_data: Raw HRM dict from load_patient()
            smooth_sigma_ms: Gaussian smoothing sigma in ms (0 = no smoothing)
            normalize: If True, normalize each sensor to [0, 1]

        Returns:
            Preprocessed HRM dict.
        """
        from scipy.ndimage import gaussian_filter1d

        channels = hrm_data['channels'].copy().astype(float)  # [T, N]
        fs = float(hrm_data['sampling_rate'])

        # Gaussian smoothing (reduces HRM pressure spikes)
        if smooth_sigma_ms > 0:
            sigma_samples = smooth_sigma_ms * fs / 1000.0
            for ch in range(channels.shape[1]):
                channels[:, ch] = gaussian_filter1d(channels[:, ch], sigma=sigma_samples)

        # Min-max normalization per sensor
        if normalize:
            ch_min = channels.min(axis=0)
            ch_max = channels.max(axis=0)
            rng = ch_max - ch_min
            rng[rng < 1e-10] = 1.0
            channels = (channels - ch_min) / rng

        preprocessed = dict(hrm_data)
        preprocessed['channels'] = channels
        preprocessed['preprocessing'] = {
            'smooth_sigma_ms': smooth_sigma_ms,
            'normalization': 'minmax' if normalize else None,
        }
        return preprocessed


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
