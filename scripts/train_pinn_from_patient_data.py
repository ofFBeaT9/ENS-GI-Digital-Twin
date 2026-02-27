"""
Train PINN and estimate patient-specific parameters from clinical files.

Expected files in patient_data/ by default:
- <PATIENT_ID>_egg.csv
- <PATIENT_ID>_hrm.csv
- <PATIENT_ID>_calcium.csv (optional)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
from ens_gi_digital.patient_data import PatientDataLoader, create_sample_patient_data  # noqa: E402


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


def run(args):
    data_dir = Path(args.data_dir)
    loader = PatientDataLoader(str(data_dir))

    egg_file = data_dir / f"{args.patient_id}_egg.csv"
    hrm_file = data_dir / f"{args.patient_id}_hrm.csv"
    if args.create_sample_if_missing and (not egg_file.exists() or not hrm_file.exists()):
        print(f"[INFO] Missing files for {args.patient_id}. Creating synthetic sample files.")
        create_sample_patient_data(
            patient_id=args.patient_id,
            output_dir=str(data_dir),
            duration_ms=args.sample_duration_ms,
            n_channels=args.sample_channels,
            sampling_rate_hz=args.sample_rate_hz,
        )

    patient = loader.load_patient_data(args.patient_id)
    if patient["voltages"] is None:
        raise ValueError("PINN estimation requires EGG voltages. Provide <PATIENT_ID>_egg.csv.")

    voltages = patient["voltages"]
    forces = patient["forces"]
    calcium = patient["calcium"]

    print("")
    print("=" * 78)
    print("PINN training and patient estimation")
    print("=" * 78)
    print(f"Patient ID: {args.patient_id}")
    print(f"Voltages shape: {voltages.shape}")
    print(f"Forces shape: {None if forces is None else forces.shape}")
    print(f"Calcium shape: {None if calcium is None else calcium.shape}")
    print("")

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
        parameter_names=[s.strip() for s in args.parameters.split(",") if s.strip()],
    )

    def train_window(duration_ms: float, epochs: int):
        dataset = pinn.generate_synthetic_dataset(
            n_samples=args.synthetic_samples,
            duration=duration_ms,
            dt=args.synthetic_dt,
            noise_level=args.synthetic_noise,
            adaptive_collocation=args.adaptive_collocation,
        )
        return pinn.train(
            features=dataset["features"],
            parameters=dataset["parameters"],
            epochs=epochs,
            verbose=1,
            generate_data=False,
            use_ode_residuals=args.use_ode_residuals,
        )

    if args.window_marching:
        window_max = args.window_max_ms if args.window_max_ms is not None else args.synthetic_duration_ms
        duration_ms = args.window_start_ms
        history = None
        stage = 1
        while duration_ms <= window_max + 1e-9:
            print("")
            print(f"[PINN] Window marching stage {stage}: duration={duration_ms:.1f} ms, epochs={args.window_epochs}")
            history = train_window(duration_ms=duration_ms, epochs=args.window_epochs)
            duration_ms += args.window_step_ms
            stage += 1
    else:
        history = train_window(duration_ms=args.synthetic_duration_ms, epochs=args.epochs)
    if history["train_loss"] and history["val_loss"]:
        print(
            "[INFO] Training done. Last losses: train={:.6f}, val={:.6f}".format(
                history["train_loss"][-1], history["val_loss"][-1]
            )
        )
    else:
        print("[INFO] Training finished with no recorded epochs.")

    estimates = pinn.estimate_parameters(
        voltages=voltages,
        forces=forces,
        calcium=calcium,
        n_bootstrap=args.bootstrap,
    )

    print("")
    print("Estimated parameters (mean +/- std):")
    for name in [s.strip() for s in args.parameters.split(",") if s.strip()]:
        if name in estimates:
            print(f"- {name}: {estimates[name]['mean']:.6f} +/- {estimates[name]['std']:.6f}")

    if args.model_out:
        pinn.save(args.model_out)

    # Apply estimates to twin and run a forward simulation for biomarker preview.
    for name, stats in estimates.items():
        set_twin_parameter(twin, name, stats["mean"])

    twin.run(duration=args.preview_duration_ms, dt=0.05, I_stim={3: 10.0}, record=True, verbose=False)
    bio = twin.extract_biomarkers()

    print("")
    print("Forward simulation biomarkers with estimated parameters:")
    print(f"- ICC frequency (cpm): {bio['icc_frequency_cpm']:.4f}")
    print(f"- Spike rate per neuron: {bio['spike_rate_per_neuron']:.4f}")
    print(f"- Motility index (%): {bio['motility_index']:.4f}")
    print(f"- Propagation correlation: {bio['propagation_correlation']:.4f}")

    print("")
    print("Done.")
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient-id", default="P001")
    parser.add_argument("--data-dir", default="patient_data")
    parser.add_argument("--segments", type=int, default=20)
    parser.add_argument("--architecture", default="mlp", choices=["mlp", "resnet"])
    parser.add_argument("--hidden-dims", default="128,64,32")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lambda-physics", type=float, default=0.1)
    parser.add_argument("--barrier-type", choices=["hinge", "exp"], default="hinge")
    parser.add_argument("--barrier-weight", type=float, default=100.0)
    parser.add_argument("--barrier-margin", type=float, default=0.02)
    parser.add_argument("--adaptive-loss-balance", action="store_true")
    parser.add_argument("--adaptive-loss-alpha", type=float, default=0.9)
    parser.add_argument("--adaptive-lambda-min", type=float, default=0.01)
    parser.add_argument("--adaptive-lambda-max", type=float, default=50.0)
    parser.add_argument("--adaptive-collocation", action="store_true")
    parser.add_argument("--collocation-focus-power", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--window-marching", action="store_true")
    parser.add_argument("--window-start-ms", type=float, default=100.0)
    parser.add_argument("--window-step-ms", type=float, default=100.0)
    parser.add_argument("--window-max-ms", type=float, default=None)
    parser.add_argument("--window-epochs", type=int, default=30)
    parser.add_argument("--synthetic-samples", type=int, default=300)
    parser.add_argument("--synthetic-duration-ms", type=float, default=600.0)
    parser.add_argument("--synthetic-dt", type=float, default=0.1)
    parser.add_argument("--synthetic-noise", type=float, default=0.05)
    parser.add_argument("--bootstrap", type=int, default=50)
    parser.add_argument("--preview-duration-ms", type=float, default=2000.0)
    parser.add_argument(
        "--parameters",
        default="g_Na,g_K,g_Ca,omega,coupling_strength",
        help="Comma-separated parameter names to estimate.",
    )
    parser.add_argument("--use-ode-residuals", action="store_true")
    parser.add_argument("--model-out", default="pinn_patient_model")
    parser.add_argument("--create-sample-if-missing", action="store_true")
    parser.add_argument("--sample-duration-ms", type=float, default=2000.0)
    parser.add_argument("--sample-channels", type=int, default=5)
    parser.add_argument("--sample-rate-hz", type=float, default=1000.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(run(args))
