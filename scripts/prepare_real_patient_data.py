"""
Prepare real-data patient CSV files for PINN training/estimation.

This script creates `patient_data/<PATIENT_ID>_egg.csv` and
`patient_data/<PATIENT_ID>_hrm.csv` using:
- Zenodo EGG dataset in `data/EGG-database`
- SPARC HRM dataset in `data/pennsieve data base/files/primary`

Output format matches PatientDataLoader.load_patient_data().
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ens_gi_digital.patient_data import PatientDataLoader  # noqa: E402


def available_hrm_subjects(data_dir: Path) -> List[int]:
    primary = data_dir / "pennsieve data base" / "files" / "primary"
    if not primary.exists():
        return []
    out = []
    for sub in primary.iterdir():
        if not sub.is_dir():
            continue
        name = sub.name.lower()
        if name.startswith("sub-"):
            try:
                out.append(int(name.split("-")[1]))
            except Exception:
                continue
    return sorted(out)


def resample_columns(source_time: np.ndarray, source_values: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    cols = []
    for idx in range(source_values.shape[1]):
        cols.append(np.interp(target_time, source_time, source_values[:, idx]))
    return np.vstack(cols).T


def match_voltage_distribution(
    voltages: np.ndarray,
    target_mean: float,
    target_std: float,
    clip_min: float,
    clip_max: float,
) -> np.ndarray:
    out = np.empty_like(voltages, dtype=np.float64)
    for i in range(voltages.shape[1]):
        x = voltages[:, i].astype(np.float64)
        x_std = float(np.std(x))
        if x_std < 1e-9:
            z = np.zeros_like(x)
        else:
            z = (x - float(np.mean(x))) / x_std
        y = z * target_std + target_mean
        out[:, i] = np.clip(y, clip_min, clip_max)
    return out


def save_patient_csv(
    out_dir: Path,
    patient_id: str,
    time_ms: np.ndarray,
    voltages: np.ndarray,
    forces: np.ndarray,
    metadata: dict,
) -> Tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    egg_cols = {f"ch{i+1}": voltages[:, i] for i in range(voltages.shape[1])}
    egg_df = pd.DataFrame({"time": time_ms, **egg_cols})
    egg_path = out_dir / f"{patient_id}_egg.csv"
    egg_df.to_csv(egg_path, index=False)

    hrm_cols = {f"sensor{i+1}": forces[:, i] for i in range(forces.shape[1])}
    hrm_df = pd.DataFrame({"time": time_ms, **hrm_cols})
    hrm_path = out_dir / f"{patient_id}_hrm.csv"
    hrm_df.to_csv(hrm_path, index=False)

    meta_path = out_dir / f"{patient_id}_meta.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return egg_path, hrm_path, meta_path


def prepare_one(
    loader: PatientDataLoader,
    out_dir: Path,
    patient_id: str,
    egg_subject: int,
    egg_condition: str,
    hrm_subject: int,
    normalize_hrm: bool,
    match_model_range: bool,
    target_vm_mean: float,
    target_vm_std: float,
    clip_vm_min: float,
    clip_vm_max: float,
    max_duration_ms: float,
) -> dict:
    egg_time, egg_voltages = loader.load_zenodo_egg(subject_id=egg_subject, condition=egg_condition)
    hrm_time, hrm_forces = loader.load_sparc_hrm(subject_id=hrm_subject, normalize=normalize_hrm)

    duration_ms = min(float(egg_time[-1]), float(hrm_time[-1]), float(max_duration_ms))
    egg_mask = egg_time <= duration_ms
    target_time = egg_time[egg_mask]
    target_voltages = egg_voltages[egg_mask]
    target_forces = resample_columns(hrm_time, hrm_forces, target_time)

    if match_model_range:
        target_voltages = match_voltage_distribution(
            target_voltages,
            target_mean=target_vm_mean,
            target_std=target_vm_std,
            clip_min=clip_vm_min,
            clip_max=clip_vm_max,
        )

    metadata = {
        "patient_id": patient_id,
        "source": {
            "egg_dataset": "Zenodo EGG (Popovic 2020)",
            "hrm_dataset": "SPARC Colonic HRM (Dinning/Brookes 2019)",
            "egg_subject": egg_subject,
            "egg_condition": egg_condition,
            "hrm_subject": hrm_subject,
        },
        "preprocessing": {
            "hrm_resampled_to_egg_time": True,
            "hrm_normalized_0_1": normalize_hrm,
            "match_model_voltage_range": match_model_range,
            "target_vm_mean_mV": target_vm_mean if match_model_range else None,
            "target_vm_std_mV": target_vm_std if match_model_range else None,
            "clip_vm_min_mV": clip_vm_min if match_model_range else None,
            "clip_vm_max_mV": clip_vm_max if match_model_range else None,
            "max_duration_ms": max_duration_ms,
        },
        "summary": {
            "n_timepoints": int(target_time.shape[0]),
            "duration_ms": float(target_time[-1] - target_time[0]) if target_time.shape[0] > 1 else 0.0,
            "egg_channels": int(target_voltages.shape[1]),
            "hrm_channels": int(target_forces.shape[1]),
            "egg_min_mV": float(np.min(target_voltages)),
            "egg_max_mV": float(np.max(target_voltages)),
            "hrm_min": float(np.min(target_forces)),
            "hrm_max": float(np.max(target_forces)),
        },
    }

    egg_path, hrm_path, meta_path = save_patient_csv(
        out_dir=out_dir,
        patient_id=patient_id,
        time_ms=target_time,
        voltages=target_voltages,
        forces=target_forces,
        metadata=metadata,
    )

    print("")
    print(f"[OK] Prepared {patient_id}")
    print(f"  EGG: {egg_path}")
    print(f"  HRM: {hrm_path}")
    print(f"  META: {meta_path}")
    print(f"  Duration: {metadata['summary']['duration_ms'] / 1000.0:.1f} s")
    print(
        "  EGG range: {:.2f} to {:.2f} mV | HRM range: {:.3f} to {:.3f}".format(
            metadata["summary"]["egg_min_mV"],
            metadata["summary"]["egg_max_mV"],
            metadata["summary"]["hrm_min"],
            metadata["summary"]["hrm_max"],
        )
    )

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="patient_data")

    parser.add_argument("--patient-id", default="R001")
    parser.add_argument("--egg-subject", type=int, default=10)
    parser.add_argument("--egg-condition", choices=["fasting", "postprandial"], default="postprandial")
    parser.add_argument("--hrm-subject", type=int, default=35)

    parser.add_argument("--batch-count", type=int, default=0, help="If >0, prepare a batch REAL001..REALNNN")
    parser.add_argument("--batch-prefix", default="REAL")
    parser.add_argument("--batch-start-index", type=int, default=1)
    parser.add_argument("--batch-egg-start", type=int, default=1)
    parser.add_argument("--batch-egg-step", type=int, default=1)
    parser.add_argument("--batch-hrm-offset", type=int, default=0, help="Offset into sorted HRM subject list")

    parser.add_argument("--keep-raw-voltage", action="store_true", help="Disable voltage distribution matching")
    parser.add_argument("--target-vm-mean", type=float, default=-65.0)
    parser.add_argument("--target-vm-std", type=float, default=3.0)
    parser.add_argument("--clip-vm-min", type=float, default=-85.0)
    parser.add_argument("--clip-vm-max", type=float, default=-40.0)

    parser.add_argument("--max-duration-ms", type=float, default=1_199_500.0)
    parser.add_argument("--hrm-raw", action="store_true", help="Use raw HRM values (default is normalized 0-1)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    loader = PatientDataLoader(str(data_dir))

    normalize_hrm = not args.hrm_raw
    match_model_range = not args.keep_raw_voltage

    if args.batch_count > 0:
        hrm_subjects = available_hrm_subjects(data_dir)
        if not hrm_subjects:
            raise FileNotFoundError("No SPARC HRM subject folders found under data/pennsieve data base/files/primary")

        print(f"[INFO] Building batch of {args.batch_count} real patients")
        print(f"[INFO] HRM subjects available: {len(hrm_subjects)}")
        for i in range(args.batch_count):
            patient_num = args.batch_start_index + i
            patient_id = f"{args.batch_prefix}{patient_num:03d}"
            egg_subject = args.batch_egg_start + i * args.batch_egg_step
            if egg_subject > 20:
                egg_subject = ((egg_subject - 1) % 20) + 1

            hrm_idx = (args.batch_hrm_offset + i) % len(hrm_subjects)
            hrm_subject = hrm_subjects[hrm_idx]

            prepare_one(
                loader=loader,
                out_dir=out_dir,
                patient_id=patient_id,
                egg_subject=egg_subject,
                egg_condition=args.egg_condition,
                hrm_subject=hrm_subject,
                normalize_hrm=normalize_hrm,
                match_model_range=match_model_range,
                target_vm_mean=args.target_vm_mean,
                target_vm_std=args.target_vm_std,
                clip_vm_min=args.clip_vm_min,
                clip_vm_max=args.clip_vm_max,
                max_duration_ms=args.max_duration_ms,
            )
    else:
        prepare_one(
            loader=loader,
            out_dir=out_dir,
            patient_id=args.patient_id,
            egg_subject=args.egg_subject,
            egg_condition=args.egg_condition,
            hrm_subject=args.hrm_subject,
            normalize_hrm=normalize_hrm,
            match_model_range=match_model_range,
            target_vm_mean=args.target_vm_mean,
            target_vm_std=args.target_vm_std,
            clip_vm_min=args.clip_vm_min,
            clip_vm_max=args.clip_vm_max,
            max_duration_ms=args.max_duration_ms,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

