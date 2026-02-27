"""
Summarize all available SPARC HRM subjects for population-level validation.

This script intentionally treats SPARC HRM as an unpaired cohort relative to
Zenodo EGG subjects.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ens_gi_digital.patient_data import PatientDataLoader  # noqa: E402


def available_subjects(data_dir: Path) -> list[int]:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--raw", action="store_true", help="Use raw mmHg instead of normalized [0,1]")
    parser.add_argument("--out-csv", default="sparc_hrm_population_summary.csv")
    parser.add_argument("--out-json", default="sparc_hrm_population_summary.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    loader = PatientDataLoader(str(data_dir))

    subjects = available_subjects(data_dir)
    if not subjects:
        raise FileNotFoundError("No SPARC subjects found in data/pennsieve data base/files/primary")

    rows = []
    for sid in subjects:
        time_ms, forces = loader.load_sparc_hrm(subject_id=sid, normalize=not args.raw)
        mean_force_trace = forces.mean(axis=1)
        dt_s = np.mean(np.diff(time_ms)) / 1000.0
        fs = 1.0 / dt_s if dt_s > 0 else 0.0

        if len(mean_force_trace) > 3 and fs > 0:
            fft = np.fft.rfft(mean_force_trace - np.mean(mean_force_trace))
            freqs = np.fft.rfftfreq(len(mean_force_trace), d=dt_s)
            idx = int(np.argmax(np.abs(fft[1:])) + 1) if len(fft) > 1 else 0
            dom_hz = float(freqs[idx]) if idx > 0 else 0.0
        else:
            dom_hz = 0.0

        rows.append(
            {
                "subject_id": sid,
                "n_timepoints": int(forces.shape[0]),
                "n_channels": int(forces.shape[1]),
                "duration_s": float((time_ms[-1] - time_ms[0]) / 1000.0),
                "mean_force": float(np.mean(forces)),
                "std_force": float(np.std(forces)),
                "p05_force": float(np.quantile(forces, 0.05)),
                "p50_force": float(np.quantile(forces, 0.50)),
                "p95_force": float(np.quantile(forces, 0.95)),
                "dominant_force_hz": dom_hz,
            }
        )

    out_csv = Path(args.out_csv)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "n_subjects": len(rows),
        "normalized": not args.raw,
        "mean_force_global": float(np.mean([r["mean_force"] for r in rows])),
        "std_force_global": float(np.mean([r["std_force"] for r in rows])),
        "mean_duration_s": float(np.mean([r["duration_s"] for r in rows])),
    }

    out_json = Path(args.out_json)
    out_json.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")

    print(f"[OK] Subjects summarized: {len(rows)}")
    print(f"[OK] Wrote: {out_csv}")
    print(f"[OK] Wrote: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

