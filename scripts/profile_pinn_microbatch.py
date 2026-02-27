"""
TensorFlow profiler micro-batch for PINN training.

This repository uses TensorFlow PINN, so this script is the equivalent of
`torch.profiler` pre-flight profiling.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import tensorflow as tf  # noqa: E402

from ens_gi_digital.core import ENSGIDigitalTwin  # noqa: E402
from ens_gi_digital.pinn import PINNConfig, PINNEstimator  # noqa: E402
from ens_gi_digital.patient_data import PatientDataLoader  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="patient_data")
    parser.add_argument("--patient-id", default="", help="Optional patient id for post-train estimate.")
    parser.add_argument("--segments", type=int, default=20)
    parser.add_argument("--architecture", choices=["mlp", "resnet"], default="mlp")
    parser.add_argument("--hidden-dims", default="128,64,32")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--synthetic-samples", type=int, default=50)
    parser.add_argument("--synthetic-duration-ms", type=float, default=400.0)
    parser.add_argument("--synthetic-dt", type=float, default=0.1)
    parser.add_argument("--synthetic-noise", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--use-ode-residuals", action="store_true")
    parser.add_argument("--adaptive-collocation", action="store_true")
    parser.add_argument("--logdir", default="tf_profile_microbatch")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("=" * 78)
    print("PINN TensorFlow Profiler Micro-Batch")
    print("=" * 78)
    print(f"TF version: {tf.__version__}")
    print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
    print(f"Profiler logdir: {args.logdir}")

    twin = ENSGIDigitalTwin(n_segments=args.segments)
    config = PINNConfig(
        architecture=args.architecture,
        hidden_dims=[int(v) for v in args.hidden_dims.split(",") if v.strip()],
        batch_size=args.batch_size,
        adaptive_collocation=args.adaptive_collocation,
    )
    pinn = PINNEstimator(
        digital_twin=twin,
        config=config,
        parameter_names=["g_Na", "g_K", "g_Ca", "omega", "coupling_strength"],
    )

    dataset = pinn.generate_synthetic_dataset(
        n_samples=args.synthetic_samples,
        duration=args.synthetic_duration_ms,
        dt=args.synthetic_dt,
        noise_level=args.synthetic_noise,
        adaptive_collocation=args.adaptive_collocation,
    )

    Path(args.logdir).mkdir(parents=True, exist_ok=True)
    tf.profiler.experimental.start(args.logdir)
    t0 = time.perf_counter()
    history = pinn.train(
        features=dataset["features"],
        parameters=dataset["parameters"],
        epochs=args.epochs,
        verbose=1,
        generate_data=False,
        use_ode_residuals=args.use_ode_residuals,
    )
    t1 = time.perf_counter()
    tf.profiler.experimental.stop()

    print("")
    print(f"[INFO] Train wall-time: {t1 - t0:.2f} s")
    if history["train_loss"]:
        print(f"[INFO] Last train loss: {history['train_loss'][-1]:.6f}")
        print(f"[INFO] Last val loss: {history['val_loss'][-1]:.6f}")

    if args.patient_id:
        loader = PatientDataLoader(args.data_dir)
        patient = loader.load_patient_data(args.patient_id)
        if patient["voltages"] is not None:
            est = pinn.estimate_parameters(
                voltages=patient["voltages"],
                forces=patient["forces"],
                calcium=patient["calcium"],
                n_bootstrap=5,
            )
            print(f"[INFO] Example estimate for {args.patient_id}:")
            for k, v in est.items():
                print(f"  - {k}: {v['mean']:.6f} +/- {v['std']:.6f}")
        else:
            print(f"[WARN] {args.patient_id} has no voltages; skipping estimate.")

    print("")
    print("[OK] Profiling complete. Open TensorBoard:")
    print(f"  tensorboard --logdir {args.logdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
