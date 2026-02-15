"""
ENS-GI Digital Twin — Complete Clinical Parameter Estimation Workflow
======================================================================
Demonstrates the full pipeline from clinical data to patient-specific parameters.

Workflow:
1. Load/generate clinical data (EGG signals, HRM measurements)
2. Train PINN on synthetic dataset (if not already trained)
3. Use PINN for fast initial parameter estimate
4. Refine estimate with Bayesian MCMC for uncertainty quantification
5. Compare both methods and generate clinical report

This represents the Phase 3 "Clinical Digital Twin" application.

Author: Mahdad
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ens_gi_digital import ENSGIDigitalTwin, IBS_PROFILES
from ens_gi_digital import PINNEstimator, PINNConfig
from ens_gi_digital import BayesianEstimator, BayesianConfig


def generate_patient_data(patient_profile: str = 'ibs_d',
                          duration: float = 3000.0,
                          noise_level: float = 0.05) -> dict:
    """Simulate patient clinical data (EGG, HRM).

    In real application, this would load actual patient data.
    """
    print(f"[Data] Generating synthetic {patient_profile} patient data...")

    twin = ENSGIDigitalTwin(n_segments=20)
    twin.apply_profile(patient_profile)

    # Run simulation
    result = twin.run(duration, dt=0.05, I_stim={5: 12.0, 6: 10.0},
                     record=True, verbose=False)

    # Add measurement noise (realistic clinical setting)
    voltages = result['voltages'] + np.random.randn(*result['voltages'].shape) * noise_level * np.std(result['voltages'])
    forces = result['force'] + np.random.randn(*result['force'].shape) * noise_level * np.std(result['force'])
    calcium = result['calcium'] + np.random.randn(*result['calcium'].shape) * noise_level * np.std(result['calcium'])

    # Get ground truth parameters (for validation)
    ground_truth = {}
    profile_obj = IBS_PROFILES[patient_profile]
    for key, val in profile_obj.membrane_mods.items():
        ground_truth[key] = val
    for key, val in profile_obj.icc_mods.items():
        ground_truth[key] = val
    for key, val in profile_obj.network_mods.items():
        ground_truth[key] = val

    return {
        'voltages': voltages,
        'forces': forces,
        'calcium': calcium,
        'ground_truth': ground_truth,
        'profile': patient_profile,
    }


def step1_pinn_estimation(patient_data: dict,
                         pretrained_model_path: str = None) -> tuple:
    """Step 1: Fast PINN-based parameter estimation."""
    print("\n" + "="*70)
    print("STEP 1: PINN Parameter Estimation (Fast, ~seconds)")
    print("="*70)

    # Create reference twin
    twin = ENSGIDigitalTwin(n_segments=20)

    # Create PINN estimator
    pinn = PINNEstimator(
        digital_twin=twin,
        config=PINNConfig(
            architecture='mlp',
            hidden_dims=[128, 64, 32],
            lambda_physics=0.1,
        ),
        parameter_names=['g_Na', 'g_K', 'g_Ca', 'omega', 'coupling_strength']
    )

    # Load pretrained model or train new one
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"[PINN] Loading pretrained model from {pretrained_model_path}")
        pinn = PINNEstimator.load(pretrained_model_path, twin)
    else:
        print("[PINN] No pretrained model found. Training new model...")
        print("[PINN] (In practice, you'd train once and reuse)")
        print("[PINN] Training on 200 synthetic samples (demo mode)...")

        # Train on small dataset for demo
        history = pinn.train(
            epochs=300,
            n_synthetic_samples=200,
            verbose=1
        )

        # Save model for future use
        pinn.save('pinn_clinical_model')
        print("[PINN] Model saved to 'pinn_clinical_model'")

    # Estimate parameters from patient data
    print("\n[PINN] Estimating parameters from patient data...")
    estimates, uncertainties = pinn.estimate_parameters(
        voltages=patient_data['voltages'],
        forces=patient_data['forces'],
        calcium=patient_data['calcium'],
        n_bootstrap=50
    )

    # Display results
    print("\n[PINN] Results:")
    print(f"{'Parameter':<20} {'Estimate':>12} {'Uncertainty':>12}")
    print("-" * 46)
    for name in pinn.parameter_names:
        print(f"{name:<20} {estimates[name]:>12.4f} ± {uncertainties[name]:>10.4f}")

    return estimates, uncertainties


def step2_bayesian_refinement(patient_data: dict,
                              pinn_estimates: dict) -> dict:
    """Step 2: Bayesian MCMC for uncertainty quantification."""
    print("\n" + "="*70)
    print("STEP 2: Bayesian Refinement (Slower, ~minutes, full uncertainty)")
    print("="*70)

    # Create reference twin
    twin = ENSGIDigitalTwin(n_segments=20)

    # Create Bayesian estimator
    bayes = BayesianEstimator(
        digital_twin=twin,
        config=BayesianConfig(
            n_chains=2,       # Use 4 for production
            n_draws=1000,     # Use 2000 for production
            n_tune=500,       # Use 1000 for production
            sampler='NUTS',
        )
    )

    # Run MCMC
    print("[Bayesian] Running MCMC sampling...")
    print("[Bayesian] (This may take 2-5 minutes...)")

    try:
        trace = bayes.estimate_parameters(
            observed_voltages=patient_data['voltages'],
            observed_forces=patient_data['forces'],
            observed_calcium=patient_data['calcium'],
            parameter_names=['g_Na', 'g_K', 'g_Ca', 'omega', 'coupling_strength']
        )

        # Get summary
        summary = bayes.summarize_posterior(trace)

        print("\n[Bayesian] Posterior Summary:")
        print(f"{'Parameter':<20} {'Mean':>12} {'95% CI':>25}")
        print("-" * 60)
        for name in summary.keys():
            if not name.startswith('sigma'):  # Skip noise parameter
                mean_val = summary[name]['mean']
                ci_low = summary[name]['ci_lower']
                ci_high = summary[name]['ci_upper']
                print(f"{name:<20} {mean_val:>12.4f} [{ci_low:>8.4f}, {ci_high:>8.4f}]")

        return summary

    except Exception as e:
        print(f"[Bayesian] Warning: MCMC failed ({e})")
        print("[Bayesian] Returning PINN estimates only")
        return None


def step3_comparison_and_report(patient_data: dict,
                               pinn_estimates: dict,
                               pinn_uncertainties: dict,
                               bayesian_summary: dict = None):
    """Step 3: Compare methods and generate clinical report."""
    print("\n" + "="*70)
    print("STEP 3: Comparison & Clinical Report")
    print("="*70)

    print("\n" + "─" * 85)
    print(f"{'Parameter':<20} {'True':>12} {'PINN':>12} {'PINN σ':>10} {'Bayes':>12} {'Bayes CI':>25}")
    print("─" * 85)

    parameter_names = ['g_Na', 'g_K', 'g_Ca', 'omega', 'coupling_strength']

    for name in parameter_names:
        # Ground truth (if available)
        true_val = patient_data['ground_truth'].get(name, np.nan)

        # PINN
        pinn_val = pinn_estimates.get(name, np.nan)
        pinn_unc = pinn_uncertainties.get(name, np.nan)

        # Bayesian
        if bayesian_summary and name in bayesian_summary:
            bayes_val = bayesian_summary[name]['mean']
            bayes_ci_low = bayesian_summary[name]['ci_lower']
            bayes_ci_high = bayesian_summary[name]['ci_upper']
            bayes_str = f"{bayes_val:>12.4f}"
            bayes_ci_str = f"[{bayes_ci_low:>8.4f}, {bayes_ci_high:>8.4f}]"
        else:
            bayes_str = "N/A"
            bayes_ci_str = "N/A"

        # Print row
        if not np.isnan(true_val):
            print(f"{name:<20} {true_val:>12.4f} {pinn_val:>12.4f} ± {pinn_unc:>8.4f} "
                  f"{bayes_str:>12} {bayes_ci_str:>25}")
        else:
            print(f"{name:<20} {'Unknown':>12} {pinn_val:>12.4f} ± {pinn_unc:>8.4f} "
                  f"{bayes_str:>12} {bayes_ci_str:>25}")

    print("─" * 85)

    # Generate clinical interpretation
    print("\n" + "═" * 70)
    print("CLINICAL INTERPRETATION")
    print("═" * 70)

    profile = patient_data['profile']
    print(f"Patient Profile: {profile.upper()}")
    print(f"Estimated Parameters:")

    # Example clinical interpretation
    g_Na_est = pinn_estimates.get('g_Na', 120)
    omega_est = pinn_estimates.get('omega', 0.005)

    if g_Na_est > 130:
        print("  • Elevated Na+ conductance → Hyperexcitability")
        print("    Recommendation: Consider Na+ channel blocker (Mexiletine)")
    elif g_Na_est < 100:
        print("  • Reduced Na+ conductance → Hypoexcitability")
        print("    Recommendation: Consider prokinetic agents")
    else:
        print("  • Normal Na+ conductance")

    omega_cpm = omega_est / (2 * np.pi) * 1000 * 60
    if omega_cpm > 5:
        print(f"  • Elevated ICC frequency ({omega_cpm:.1f} cpm) → Rapid transit")
    elif omega_cpm < 2:
        print(f"  • Reduced ICC frequency ({omega_cpm:.1f} cpm) → Slow transit")
    else:
        print(f"  • Normal ICC frequency ({omega_cpm:.1f} cpm)")

    print("\n" + "═" * 70)


def main():
    """Run complete workflow."""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  ENS-GI Clinical Parameter Estimation — Complete Workflow    ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # Choose patient profile for demo
    patient_profile = 'ibs_d'  # Can be 'healthy', 'ibs_d', 'ibs_c', 'ibs_m'

    # Step 0: Generate/load patient data
    patient_data = generate_patient_data(patient_profile=patient_profile)

    # Step 1: PINN estimation
    pinn_estimates, pinn_uncertainties = step1_pinn_estimation(patient_data)

    # Step 2: Bayesian refinement (optional but recommended)
    bayesian_summary = step2_bayesian_refinement(patient_data, pinn_estimates)

    # Step 3: Comparison and clinical report
    step3_comparison_and_report(patient_data, pinn_estimates, pinn_uncertainties,
                               bayesian_summary)

    print("\n✓ Workflow complete!")
    print("\nNEXT STEPS:")
    print("  1. Apply estimated parameters to patient-specific digital twin")
    print("  2. Run virtual drug trials to predict treatment response")
    print("  3. Generate personalized treatment recommendations")
    print("  4. Monitor patient over time and update parameters")


if __name__ == '__main__':
    main()
