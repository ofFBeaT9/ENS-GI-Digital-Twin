"""
Train one PINN model, then estimate parameters for all Zenodo EGG subjects.

Scientific use:
- Uses all 20 real EGG subjects (same modality, no cross-subject pairing).
- Produces per-subject parameter estimates + forward biomarkers.

Note:
- This script does NOT pair SPARC HRM individuals with Zenodo EGG individuals.
  Use HRM separately for population-level validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List

import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ens_gi_digital.core import ENSGIDigitalTwin  # noqa: E402
from ens_gi_digital.pinn import PINNConfig, PINNEstimator  # noqa: E402
from ens_gi_digital.patient_data import PatientDataLoader  # noqa: E402


def parse_conditions(raw: str) -> List[str]:
    """Parse a comma-separated condition string, e.g. 'fasting,postprandial'."""
    valid = {"fasting", "postprandial"}
    out = [c.strip() for c in raw.split(",") if c.strip()]
    for c in out:
        if c not in valid:
            raise ValueError(f"Unknown condition: {c!r}. Valid: {sorted(valid)}")
    return out


def parse_subjects(raw: str) -> List[int]:
    if not raw.strip():
        return list(range(1, 21))
    out = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            start = int(a.strip())
            end = int(b.strip())
            step = 1 if start <= end else -1
            out.extend(list(range(start, end + step, step)))
        else:
            out.append(int(token))
    return sorted(set([s for s in out if 1 <= s <= 20]))


def set_twin_parameter(twin: ENSGIDigitalTwin, name: str, value: float) -> bool:
    if hasattr(twin.network.neurons[0].params, name):
        for neuron in twin.network.neurons:
            setattr(neuron.params, name, value)
        return True
    if hasattr(twin.network.params, name):
        setattr(twin.network.params, name, value)
        return True
    if hasattr(twin.icc.params, name):
        setattr(twin.icc.params, name, value)
        return True
    if hasattr(twin.muscle.params, name):
        setattr(twin.muscle.params, name, value)
        return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--conditions",
        default="fasting,postprandial",
        help="Comma-separated list of conditions to estimate. "
             "E.g.: 'fasting,postprandial' (default) or just 'postprandial'.")
    # --condition (singular) kept for backward compatibility; overrides --conditions
    parser.add_argument(
        "--condition",
        choices=["fasting", "postprandial"],
        default=None,
        help="Single condition (overrides --conditions if specified).")
    parser.add_argument("--subjects", default="1-20", help="Examples: 1-20 or 1,2,5,8")

    parser.add_argument("--segments", type=int, default=20)
    parser.add_argument("--architecture", choices=["mlp", "resnet"], default="resnet")
    parser.add_argument("--hidden-dims", default="512,256,128,64,32")
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--lambda-physics", type=float, default=0.2)
    parser.add_argument("--barrier-type", choices=["hinge", "exp"], default="hinge")
    parser.add_argument("--barrier-weight", type=float, default=100.0)
    parser.add_argument("--barrier-margin", type=float, default=0.02)
    parser.add_argument("--adaptive-loss-balance", action="store_true")
    parser.add_argument("--adaptive-loss-alpha", type=float, default=0.9)
    parser.add_argument("--adaptive-lambda-min", type=float, default=0.01)
    parser.add_argument("--adaptive-lambda-max", type=float, default=50.0)
    parser.add_argument("--adaptive-collocation", action="store_true")
    parser.add_argument("--collocation-focus-power", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--window-marching", action="store_true")
    parser.add_argument("--window-start-ms", type=float, default=100.0)
    parser.add_argument("--window-step-ms", type=float, default=100.0)
    parser.add_argument("--window-max-ms", type=float, default=None)
    parser.add_argument("--window-epochs", type=int, default=30)
    parser.add_argument("--synthetic-samples", type=int, default=800)
    parser.add_argument("--synthetic-duration-ms", type=float, default=1000.0)
    parser.add_argument("--synthetic-dt", type=float, default=0.1)
    parser.add_argument("--synthetic-noise", type=float, default=0.05)
    parser.add_argument("--sim-batch-size", type=int, default=500,
                        help="Simulations per parallel batch (lower = less CPU pressure).")
    parser.add_argument("--bootstrap", type=int, default=120)
    parser.add_argument("--preview-duration-ms", type=float, default=2000.0)
    parser.add_argument("--use-ode-residuals", action="store_true")

    parser.add_argument("--model-out", default="pinn_zenodo_cohort")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Directory for per-stage checkpoints (default: <model-out>_stages).")
    parser.add_argument("--results-csv", default="pinn_zenodo_cohort_results.csv")
    parser.add_argument("--results-json", default="pinn_zenodo_cohort_results.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    subjects = parse_subjects(args.subjects)
    if not subjects:
        raise ValueError("No valid subject IDs selected.")

    # Resolve conditions: --condition (singular) overrides --conditions
    if args.condition is not None:
        conditions = [args.condition]
    else:
        conditions = parse_conditions(args.conditions)
    if not conditions:
        raise ValueError("No valid conditions selected.")

    print("=" * 78)
    print("PINN cohort training on Zenodo EGG")
    print("=" * 78)
    print(f"Conditions: {', '.join(conditions)}")
    print(f"Subjects:   {subjects}")
    print("")

    loader = PatientDataLoader(args.data_dir)
    twin = ENSGIDigitalTwin(n_segments=args.segments)
    config = PINNConfig(
        architecture=args.architecture,
        hidden_dims=[int(v) for v in args.hidden_dims.split(",") if v.strip()],
        learning_rate=args.learning_rate,
        lambda_physics=args.lambda_physics,
        batch_size=args.batch_size,
        barrier_type=args.barrier_type,
        barrier_weight=args.barrier_weight,
        barrier_margin=args.barrier_margin,
        adaptive_loss_balance=args.adaptive_loss_balance,
        adaptive_loss_alpha=args.adaptive_loss_alpha,
        adaptive_lambda_min=args.adaptive_lambda_min,
        adaptive_lambda_max=args.adaptive_lambda_max,
        adaptive_collocation=args.adaptive_collocation,
        collocation_focus_power=args.collocation_focus_power,
    )
    pinn = PINNEstimator(
        digital_twin=twin,
        config=config,
        parameter_names=["g_Na", "g_K", "g_Ca", "omega", "coupling_strength"],
    )

    def train_window(duration_ms: float, epochs: int):
        dataset = pinn.generate_synthetic_dataset(
            n_samples=args.synthetic_samples,
            duration=duration_ms,
            dt=args.synthetic_dt,
            noise_level=args.synthetic_noise,
            adaptive_collocation=args.adaptive_collocation,
            sim_batch_size=args.sim_batch_size,
        )
        return pinn.train(
            features=dataset["features"],
            parameters=dataset["parameters"],
            epochs=epochs,
            verbose=1,
            generate_data=False,
            use_ode_residuals=args.use_ode_residuals,
        )

    ckpt_dir = Path(args.checkpoint_dir if args.checkpoint_dir else f"{args.model_out}_stages")

    if args.window_marching:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        window_max = args.window_max_ms if args.window_max_ms is not None else args.synthetic_duration_ms
        duration_ms = args.window_start_ms
        history = None
        stage = 1
        while duration_ms <= window_max + 1e-9:
            stage_tag = f"stage_{stage:03d}_{duration_ms:.0f}ms"
            ckpt_path = str(ckpt_dir / stage_tag)
            ckpt_keras = ckpt_dir / f"{stage_tag}.keras"

            print("")
            print(f"[PINN] Window marching stage {stage}: duration={duration_ms:.1f} ms, epochs={args.window_epochs}")

            if ckpt_keras.exists():
                print(f"[PINN] Checkpoint found — loading {ckpt_path}")
                pinn = PINNEstimator.load(ckpt_path, twin)
            else:
                _cuda_exc = (tf.errors.InternalError,) if tf is not None else ()
                _catch = _cuda_exc + (RuntimeError,)
                try:
                    history = train_window(duration_ms=duration_ms, epochs=args.window_epochs)
                except _catch as exc:
                    print(f"\n[PINN] CUDA crash at stage {stage} ({duration_ms:.0f} ms): {exc}")
                    print(f"[PINN] Re-run the same command to resume automatically from stage {stage}.")
                    raise SystemExit(1)
                pinn.save(ckpt_path)
                print(f"[PINN] Stage checkpoint saved: {ckpt_path}")

            duration_ms += args.window_step_ms
            stage += 1
    else:
        history = train_window(duration_ms=args.synthetic_duration_ms, epochs=args.epochs)
    if history is not None and history["train_loss"] and history["val_loss"]:
        print(
            "[INFO] Training done. Last losses: train={:.6f}, val={:.6f}".format(
                history["train_loss"][-1], history["val_loss"][-1]
            )
        )
    pinn.save(args.model_out)

    rows = []
    for sid in subjects:
        for condition in conditions:
            print("")
            print(f"[INFO] Estimating subject ID{sid} ({condition})")
            try:
                _, voltages = loader.load_zenodo_egg(subject_id=sid, condition=condition)
            except Exception as exc:
                print(f"[WARN] Subject {sid} {condition}: {exc} — skipping")
                continue
            estimates = pinn.estimate_parameters(
                voltages=voltages,
                forces=None,
                calcium=None,
                n_bootstrap=args.bootstrap,
            )

            preview_twin = ENSGIDigitalTwin(n_segments=args.segments)
            for param_name, stats in estimates.items():
                set_twin_parameter(preview_twin, param_name, stats["mean"])
            preview_twin.run(
                duration=args.preview_duration_ms,
                dt=0.05,
                I_stim={3: 10.0},
                record=True,
                verbose=False,
            )
            bio = preview_twin.extract_biomarkers()

            row = {
                "subject_id": sid,
                "condition": condition,
                "g_Na_mean": estimates["g_Na"]["mean"],
                "g_Na_std": estimates["g_Na"]["std"],
                "g_K_mean": estimates["g_K"]["mean"],
                "g_K_std": estimates["g_K"]["std"],
                "g_Ca_mean": estimates["g_Ca"]["mean"],
                "g_Ca_std": estimates["g_Ca"]["std"],
                "omega_mean": estimates["omega"]["mean"],
                "omega_std": estimates["omega"]["std"],
                "coupling_mean": estimates["coupling_strength"]["mean"],
                "coupling_std": estimates["coupling_strength"]["std"],
                "icc_frequency_cpm": bio["icc_frequency_cpm"],
                "spike_rate_per_neuron": bio["spike_rate_per_neuron"],
                "motility_index": bio["motility_index"],
                "propagation_correlation": bio["propagation_correlation"],
            }
            rows.append(row)

    if not rows:
        raise RuntimeError("No rows generated — all subjects/conditions failed.")

    csv_path = Path(args.results_csv)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ── Per-condition summary statistics ──────────────────────────────────────
    conditions_done = sorted(set(r["condition"] for r in rows))
    per_condition: dict = {}
    for cond in conditions_done:
        cond_rows = [r for r in rows if r["condition"] == cond]
        per_condition[cond] = {
            "n_subjects": len(cond_rows),
            "mean_motility_index":    float(np.mean([r["motility_index"] for r in cond_rows])),
            "mean_icc_frequency_cpm": float(np.mean([r["icc_frequency_cpm"] for r in cond_rows])),
            "mean_g_Na":      float(np.mean([r["g_Na_mean"] for r in cond_rows])),
            "mean_g_K":       float(np.mean([r["g_K_mean"] for r in cond_rows])),
            "mean_g_Ca":      float(np.mean([r["g_Ca_mean"] for r in cond_rows])),
            "mean_omega":     float(np.mean([r["omega_mean"] for r in cond_rows])),
            "mean_coupling":  float(np.mean([r["coupling_mean"] for r in cond_rows])),
        }

    summary = {
        "subjects":        subjects,
        "conditions":      conditions_done,
        "n_subjects":      len(subjects),
        "n_rows":          len(rows),
        "per_condition":   per_condition,
        "results_csv":     str(csv_path),
    }

    json_path = Path(args.results_json)
    json_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")

    print("")
    print("=" * 78)
    print("Cohort estimation complete")
    print("=" * 78)
    print(f"Subjects processed: {len(subjects)}")
    print(f"Total rows:         {len(rows)}")
    for cond, stats in per_condition.items():
        print(f"\n  [{cond}]  n={stats['n_subjects']}")
        print(f"    Mean ICC frequency:  {stats['mean_icc_frequency_cpm']:.4f} cpm")
        print(f"    Mean motility index: {stats['mean_motility_index']:.4f}")
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
