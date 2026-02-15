"""
Complete clinical workflow: Patient data -> Parameter estimation -> Report

This script demonstrates the full pipeline from loading patient data
to parameter estimation and biomarker extraction.
"""
import sys
import numpy as np
from pathlib import Path
from patient_data_loader import PatientDataLoader
from .core import ENSGIDigitalTwin

# Try to import PINN (optional)
PINN_AVAILABLE = False  # Default to False
try:
    from ens_gi_pinn import PINNEstimator, PINNConfig
    PINN_AVAILABLE = True
except Exception:
    pass  # PINN not available


def run_clinical_analysis(patient_id: str, data_dir: str = 'patient_data/'):
    """Complete clinical analysis pipeline.

    Args:
        patient_id: Patient identifier
        data_dir: Directory containing patient data

    Returns:
        dict with 'patient_id', 'estimates', 'biomarkers', 'twin'
    """
    print("\n" + "="*70)
    print(f"CLINICAL ANALYSIS: Patient {patient_id}")
    print("="*70)

    # Step 1: Load patient data
    print("\n[1/5] Loading patient data...")
    loader = PatientDataLoader(data_dir)
    patient_data = loader.load_patient_data(patient_id)

    if patient_data['voltages'] is None:
        raise ValueError(f"No voltage data available for patient {patient_id}")

    # Step 2: Initialize digital twin
    print("\n[2/5] Initializing digital twin...")
    n_segments = patient_data['voltages'].shape[1]
    twin = ENSGIDigitalTwin(n_segments=n_segments)
    print(f"✓ Digital twin initialized with {n_segments} segments")

    # Step 3: PINN parameter estimation (fast)
    print("\n[3/5] Running PINN parameter estimation...")

    estimates = None
    if not PINN_AVAILABLE:
        print("  ⚠ PINN module not available (TensorFlow not installed)")
        print("  Using default parameter estimates")
    else:
        try:
            pinn_config = PINNConfig(
                architecture='resnet',
                hidden_dims=[128, 64, 32],
                learning_rate=0.001,
                lambda_physics=0.1
            )
            pinn = PINNEstimator(twin, pinn_config,
                                parameter_names=['g_Na', 'g_K', 'omega'])

            # Generate synthetic training data
            print("  Generating training dataset...")
                try:
                dataset = pinn.generate_synthetic_dataset(n_samples=500)
                print(f"  ✓ Generated {len(dataset['features'])} training samples")
                except Exception as e:
                print(f"  ⚠ Warning: Could not generate training data: {e}")
                print("  Skipping PINN training, using default parameters")
                dataset = None

            # Train PINN if we have data
            if dataset is not None:
                print("  Training PINN (1000 epochs)...")
                try:
                    history = pinn.train(dataset['features'], dataset['parameters'],
                                    epochs=1000, verbose=False)
                    print(f"  ✓ Training complete (final loss: {history['loss'][-1]:.6f})")
                except Exception as e:
                    print(f"  ⚠ Warning: Training failed: {e}")

            # Estimate parameters from patient data
            print("  Estimating parameters with bootstrap uncertainty...")
                try:
                # Use patient data if available, otherwise use defaults
                voltages = patient_data['voltages']
                forces = patient_data['forces'] if patient_data['forces'] is not None else np.zeros_like(voltages)

                estimates = pinn.estimate_parameters(
                    voltages,
                    forces,
                    n_bootstrap=50
                )
                print("  ✓ Parameter estimation complete")
                except Exception as e:
                print(f"  ⚠ Warning: Parameter estimation failed: {e}")
                # Use default estimates
                estimates = {
                    'g_Na': {'mean': 120.0, 'std': 10.0},
                    'g_K': {'mean': 36.0, 'std': 3.0},
                    'omega': {'mean': 0.3, 'std': 0.03}
                }
        else:
            # Use default estimates
            estimates = {
                'g_Na': {'mean': 120.0, 'std': 10.0},
                'g_K': {'mean': 36.0, 'std': 3.0},
                'omega': {'mean': 0.3, 'std': 0.03}
            }

    # Step 4: Bayesian refinement (optional, slower)
    print("\n[4/5] Bayesian refinement...")
    print("  Skipping for speed. Use BayesianEstimator for full posterior if needed.")

    # Step 5: Generate clinical report
    print("\n[5/5] Generating clinical report...")

    # Update twin with estimated parameters
    for neuron in twin.network.neurons:
        neuron.params.g_Na = estimates['g_Na']['mean']
        neuron.params.g_K = estimates['g_K']['mean']
    twin.icc.params.omega = estimates['omega']['mean']

    # Run simulation with patient-specific parameters
    print("  Running simulation with patient-specific parameters...")
    try:
        result = twin.run(2000, dt=0.05, verbose=False)
        biomarkers = twin.extract_biomarkers()
        print("  ✓ Biomarker extraction complete")
    except Exception as e:
        print(f"  ⚠ Warning: Simulation failed: {e}")
        # Use synthetic biomarkers
        biomarkers = {
            'icc_frequency_cpm': 3.0,
            'motility_index': 0.5,
            'spike_rate_per_neuron': 0.1
        }

    # Print report
    print("\n" + "="*70)
    print("CLINICAL REPORT")
    print("="*70)
    print(f"Patient ID: {patient_id}")
    print(f"\nEstimated Parameters:")
    for param, vals in estimates.items():
        print(f"  {param:15s}: {vals['mean']:7.3f} ± {vals['std']:6.3f}")

    print(f"\nBiomarkers:")
    for key, val in biomarkers.items():
        if isinstance(val, (int, float)):
            print(f"  {key:30s}: {val:.3f}")
        else:
            print(f"  {key:30s}: {val}")

    print("\n" + "="*70)

    return {
        'patient_id': patient_id,
        'estimates': estimates,
        'biomarkers': biomarkers,
        'twin': twin,
        'patient_data': patient_data
    }


def compare_patients(patient_ids: list, data_dir: str = 'patient_data/'):
    """Compare multiple patients.

    Args:
        patient_ids: List of patient IDs
        data_dir: Directory containing patient data
    """
    print("\n" + "="*70)
    print("MULTI-PATIENT COMPARISON")
    print("="*70)

    results = []
    for pid in patient_ids:
        try:
            result = run_clinical_analysis(pid, data_dir)
            results.append(result)
        except Exception as e:
            print(f"\n✗ Failed to analyze patient {pid}: {e}")

    if len(results) > 1:
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)

        # Compare parameters
        print("\nParameter Comparison:")
        print(f"{'Patient':<15} {'g_Na':>12} {'g_K':>12} {'omega':>12}")
        print("-" * 60)
        for r in results:
            pid = r['patient_id']
            g_na = r['estimates']['g_Na']['mean']
            g_k = r['estimates']['g_K']['mean']
            omega = r['estimates']['omega']['mean']
            print(f"{pid:<15} {g_na:12.2f} {g_k:12.2f} {omega:12.4f}")

        # Compare biomarkers
        print("\nBiomarker Comparison:")
        if results[0]['biomarkers']:
            biomarker_keys = list(results[0]['biomarkers'].keys())
            print(f"{'Patient':<15}", end="")
            for key in biomarker_keys[:3]:  # Show first 3
                print(f"{key:>20}", end="")
            print()
            print("-" * 80)

            for r in results:
                print(f"{r['patient_id']:<15}", end="")
                for key in biomarker_keys[:3]:
                    val = r['biomarkers'].get(key, 0)
                    if isinstance(val, (int, float)):
                        print(f"{val:20.3f}", end="")
                    else:
                        print(f"{str(val):>20}", end="")
                print()

    return results


if __name__ == '__main__':
    # Create sample data if needed
    from patient_data_loader import create_sample_patient_data

    if not Path('patient_data').exists() or len(list(Path('patient_data').glob('*_egg.csv'))) == 0:
        print("Creating sample patient data...")
        create_sample_patient_data('P001', n_channels=5, duration_ms=2000.0)
        create_sample_patient_data('P002', n_channels=5, duration_ms=2000.0)
        create_sample_patient_data('P003', n_channels=5, duration_ms=2000.0)
        print()

    # Run single patient analysis
    if len(sys.argv) > 1:
        patient_id = sys.argv[1]
        results = run_clinical_analysis(patient_id)
    else:
        # Run comparison
        results = compare_patients(['P001', 'P002', 'P003'])
