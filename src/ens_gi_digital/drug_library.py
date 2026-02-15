"""
ENS-GI Digital Twin — Drug Library & Virtual Drug Trials
=========================================================
Enables in silico drug testing and treatment optimization.

Key Features:
- Comprehensive drug library (Mexiletine, Ondansetron, Linaclotide, etc.)
- Pharmacokinetic modeling (ADME: Absorption, Distribution, Metabolism, Excretion)
- Pharmacodynamic modeling (dose-response curves, IC50/EC50)
- Virtual clinical trials (cohort generation, statistical analysis)
- Multi-drug interaction modeling

Clinical Applications:
- Predict patient response before prescribing
- Optimize dosage personalized to patient parameters
- Test combination therapies
- Identify contraindications

Usage:
    from .core import ENSGIDigitalTwin
    from ens_gi_drug_library import DrugLibrary, apply_drug, VirtualDrugTrial

    # Apply drug to digital twin
    twin = ENSGIDigitalTwin(n_segments=20)
    twin.apply_profile('ibs_c')

    # Test Mexiletine
    apply_drug(twin, DrugLibrary.MEXILETINE, dose_mg=200)
    result = twin.run(duration=2000)

    # Virtual trial
    trial = VirtualDrugTrial(
        drug=DrugLibrary.MEXILETINE,
        cohort_size=100,
        patient_profile='ibs_c'
    )
    trial_results = trial.run()

Author: Mahdad (Phase 3 Implementation)
License: MIT
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
    sns.set_style('whitegrid')
except ImportError:
    PLOTTING_AVAILABLE = False

# Import core digital twin
from .core import ENSGIDigitalTwin, MembraneParams, NetworkParams, ICCParams


# ═══════════════════════════════════════════════════════════════
# Drug Definitions
# ═══════════════════════════════════════════════════════════════

class TargetType(Enum):
    """Types of drug targets."""
    ION_CHANNEL = "ion_channel"
    RECEPTOR = "receptor"
    ENZYME = "enzyme"
    TRANSPORTER = "transporter"
    MODULATOR = "modulator"


@dataclass
class DrugTarget:
    """Single drug target specification."""
    parameter: str              # Parameter name (e.g., 'g_Na', 'serotonin_factor')
    target_type: TargetType
    effect_function: Callable[[float, float], float]  # (baseline_value, concentration) -> modified_value
    ic50_or_ec50: float        # Half-maximal concentration (μM)
    hill_coefficient: float = 1.0  # Hill coefficient for dose-response
    efficacy: float = 1.0      # Maximum effect (0-1 for partial agonists)


@dataclass
class DrugProfile:
    """Complete drug specification."""
    name: str
    generic_name: str
    drug_class: str
    indication: str
    targets: List[DrugTarget]

    # Pharmacokinetics (PK)
    bioavailability: float = 1.0   # Fraction absorbed (0-1)
    half_life_hours: float = 4.0   # Plasma half-life
    volume_distribution_L: float = 50.0  # Volume of distribution

    # Administration
    standard_dose_mg: float = 100.0
    max_dose_mg: float = 400.0

    # Side effects (optional)
    side_effects: Dict[str, float] = field(default_factory=dict)
    contraindications: List[str] = field(default_factory=list)

    # Notes
    mechanism: str = ""
    references: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# Drug Library
# ═══════════════════════════════════════════════════════════════

class DrugLibrary:
    """Library of GI-relevant drugs for virtual trials."""

    # Sodium channel blocker - IBS-C rescue
    MEXILETINE = DrugProfile(
        name="Mexiletine",
        generic_name="mexiletine",
        drug_class="Class IB antiarrhythmic / Na+ channel blocker",
        indication="IBS-C, gastroparesis, neuropathic pain",
        targets=[
            DrugTarget(
                parameter='g_Na',
                target_type=TargetType.ION_CHANNEL,
                effect_function=lambda baseline, conc: baseline * (1 - 0.6 * conc / (conc + 10.0)),
                ic50_or_ec50=10.0,  # μM
                hill_coefficient=1.2,
                efficacy=0.6  # 60% block at saturation
            )
        ],
        bioavailability=0.85,
        half_life_hours=10.0,
        volume_distribution_L=70.0,
        standard_dose_mg=200.0,
        max_dose_mg=400.0,
        mechanism="Blocks voltage-gated Na+ channels, reduces hyperexcitability",
        references=["Parkman et al. 2013, Digestive Diseases"]
    )

    # 5-HT3 antagonist - IBS-D
    ONDANSETRON = DrugProfile(
        name="Ondansetron",
        generic_name="ondansetron",
        drug_class="5-HT3 receptor antagonist",
        indication="IBS-D, nausea, chemotherapy-induced emesis",
        targets=[
            DrugTarget(
                parameter='serotonin_factor',
                target_type=TargetType.RECEPTOR,
                effect_function=lambda baseline, conc: baseline * (1 - 0.5 * conc / (conc + 2.0)),
                ic50_or_ec50=2.0,  # μM
                hill_coefficient=1.0,
                efficacy=0.5
            )
        ],
        bioavailability=0.6,
        half_life_hours=3.5,
        volume_distribution_L=140.0,
        standard_dose_mg=8.0,
        max_dose_mg=16.0,
        mechanism="Blocks 5-HT3 receptors, reduces serotonin-mediated hypermotility",
        references=["Garsed et al. 2014, Gut"]
    )

    # 5-HT3 antagonist - IBS-D (more potent)
    ALOSETRON = DrugProfile(
        name="Alosetron",
        generic_name="alosetron",
        drug_class="5-HT3 receptor antagonist",
        indication="Severe IBS-D (women only, FDA restricted)",
        targets=[
            DrugTarget(
                parameter='serotonin_factor',
                target_type=TargetType.RECEPTOR,
                effect_function=lambda baseline, conc: baseline * (1 - 0.7 * conc / (conc + 0.5)),
                ic50_or_ec50=0.5,  # μM (more potent than ondansetron)
                hill_coefficient=1.1,
                efficacy=0.7
            ),
            DrugTarget(
                parameter='excitatory_weight',
                target_type=TargetType.MODULATOR,
                effect_function=lambda baseline, conc: baseline * (1 - 0.3 * conc / (conc + 1.0)),
                ic50_or_ec50=1.0,
                hill_coefficient=1.0,
                efficacy=0.3
            )
        ],
        bioavailability=0.5,
        half_life_hours=1.5,
        volume_distribution_L=65.0,
        standard_dose_mg=1.0,
        max_dose_mg=2.0,
        mechanism="Potent 5-HT3 antagonism, reduces motility and visceral hypersensitivity",
        contraindications=["Ischemic colitis risk", "Severe constipation"],
        references=["Camilleri et al. 2000, Gastroenterology"]
    )

    # ClC-2 activator - IBS-C
    LUBIPROSTONE = DrugProfile(
        name="Lubiprostone",
        generic_name="lubiprostone",
        drug_class="Chloride channel activator",
        indication="IBS-C, chronic constipation",
        targets=[
            DrugTarget(
                parameter='g_L',  # Leak conductance (approximation)
                target_type=TargetType.ION_CHANNEL,
                effect_function=lambda baseline, conc: baseline * (1 + 0.8 * conc / (conc + 5.0)),
                ic50_or_ec50=5.0,  # μM (EC50 for activation)
                hill_coefficient=1.3,
                efficacy=0.8
            ),
            DrugTarget(
                parameter='omega',  # Indirect ICC enhancement
                target_type=TargetType.MODULATOR,
                effect_function=lambda baseline, conc: baseline * (1 + 0.2 * conc / (conc + 10.0)),
                ic50_or_ec50=10.0,
                hill_coefficient=1.0,
                efficacy=0.2
            )
        ],
        bioavailability=0.001,  # Very low (local action)
        half_life_hours=0.9,
        volume_distribution_L=100.0,
        standard_dose_mg=0.024,  # 24 μg
        max_dose_mg=0.048,
        mechanism="Activates ClC-2 chloride channels, increases intestinal fluid secretion",
        side_effects={'nausea': 0.3, 'diarrhea': 0.1},
        references=["Drossman et al. 2009, Gastroenterology"]
    )

    # Guanylate cyclase agonist - IBS-C
    LINACLOTIDE = DrugProfile(
        name="Linaclotide",
        generic_name="linaclotide",
        drug_class="Guanylate cyclase-C agonist",
        indication="IBS-C, chronic constipation",
        targets=[
            DrugTarget(
                parameter='coupling_strength',
                target_type=TargetType.MODULATOR,
                effect_function=lambda baseline, conc: baseline * (1 + 0.5 * conc / (conc + 3.0)),
                ic50_or_ec50=3.0,  # μM
                hill_coefficient=1.4,
                efficacy=0.5
            ),
            DrugTarget(
                parameter='omega',
                target_type=TargetType.MODULATOR,
                effect_function=lambda baseline, conc: baseline * (1 + 0.3 * conc / (conc + 5.0)),
                ic50_or_ec50=5.0,
                hill_coefficient=1.2,
                efficacy=0.3
            )
        ],
        bioavailability=0.0,  # Minimal systemic absorption (gut-restricted)
        half_life_hours=0.0,  # Local action
        volume_distribution_L=1.0,
        standard_dose_mg=0.29,  # 290 μg
        max_dose_mg=0.58,
        mechanism="Increases cGMP, enhances chloride/bicarbonate secretion and transit",
        side_effects={'diarrhea': 0.2},
        references=["Rao et al. 2012, Gastroenterology"]
    )

    # 5-HT4 agonist - Prokinetic
    PRUCALOPRIDE = DrugProfile(
        name="Prucalopride",
        generic_name="prucalopride",
        drug_class="5-HT4 receptor agonist",
        indication="Chronic constipation, gastroparesis",
        targets=[
            DrugTarget(
                parameter='excitatory_weight',
                target_type=TargetType.RECEPTOR,
                effect_function=lambda baseline, conc: baseline * (1 + 0.6 * conc / (conc + 2.0)),
                ic50_or_ec50=2.0,  # μM (EC50)
                hill_coefficient=1.3,
                efficacy=0.6
            ),
            DrugTarget(
                parameter='coupling_strength',
                target_type=TargetType.MODULATOR,
                effect_function=lambda baseline, conc: baseline * (1 + 0.4 * conc / (conc + 3.0)),
                ic50_or_ec50=3.0,
                hill_coefficient=1.0,
                efficacy=0.4
            )
        ],
        bioavailability=0.9,
        half_life_hours=24.0,  # Long half-life (once daily)
        volume_distribution_L=567.0,
        standard_dose_mg=2.0,
        max_dose_mg=4.0,
        mechanism="Selective 5-HT4 agonism, enhances cholinergic neurotransmission and motility",
        references=["Camilleri et al. 2008, Gastroenterology"]
    )

    # Antibiotic with neuromodulatory effects
    RIFAXIMIN = DrugProfile(
        name="Rifaximin",
        generic_name="rifaximin",
        drug_class="Non-absorbable antibiotic",
        indication="IBS-D, hepatic encephalopathy, SIBO",
        targets=[
            DrugTarget(
                parameter='serotonin_factor',
                target_type=TargetType.MODULATOR,
                effect_function=lambda baseline, conc: baseline * (1 - 0.3 * conc / (conc + 10.0)),
                ic50_or_ec50=10.0,
                hill_coefficient=1.0,
                efficacy=0.3
            ),
            DrugTarget(
                parameter='inhibitory_weight',
                target_type=TargetType.MODULATOR,
                effect_function=lambda baseline, conc: baseline * (1 + 0.2 * conc / (conc + 15.0)),
                ic50_or_ec50=15.0,
                hill_coefficient=1.0,
                efficacy=0.2
            )
        ],
        bioavailability=0.004,  # <1% absorbed (gut-restricted)
        half_life_hours=6.0,
        volume_distribution_L=5.0,
        standard_dose_mg=550.0,
        max_dose_mg=1650.0,  # 550 mg TID
        mechanism="Reduces bacterial overgrowth, modulates gut microbiome-ENS interactions",
        references=["Pimentel et al. 2011, NEJM"]
    )

    @classmethod
    def get_all_drugs(cls) -> List[DrugProfile]:
        """Get list of all available drugs."""
        return [
            cls.MEXILETINE,
            cls.ONDANSETRON,
            cls.ALOSETRON,
            cls.LUBIPROSTONE,
            cls.LINACLOTIDE,
            cls.PRUCALOPRIDE,
            cls.RIFAXIMIN,
        ]

    @classmethod
    def get_by_name(cls, name: str) -> Optional[DrugProfile]:
        """Get drug by name."""
        for drug in cls.get_all_drugs():
            if drug.name.lower() == name.lower() or drug.generic_name.lower() == name.lower():
                return drug
        return None


# ═══════════════════════════════════════════════════════════════
# Pharmacokinetics & Pharmacodynamics
# ═══════════════════════════════════════════════════════════════

def compute_plasma_concentration(dose_mg: float,
                                 drug: DrugProfile,
                                 time_hours: float) -> float:
    """Compute plasma concentration using one-compartment model.

    Args:
        dose_mg: Oral dose in milligrams
        drug: DrugProfile with PK parameters
        time_hours: Time since administration

    Returns:
        Plasma concentration in μM
    """
    # Convert dose to μmol (assuming average MW ~300 g/mol)
    molecular_weight = 300.0  # g/mol (approximate)
    dose_umol = (dose_mg / molecular_weight) * 1000.0  # μmol

    # Peak concentration (Cmax)
    C_max = (dose_umol * drug.bioavailability) / drug.volume_distribution_L  # μM

    # Elimination rate constant
    k_e = 0.693 / drug.half_life_hours  # 1/hours

    # Concentration at time t (exponential decay)
    C_t = C_max * np.exp(-k_e * time_hours)

    return C_t


def apply_drug_effect(baseline_value: float,
                     concentration_uM: float,
                     target: DrugTarget) -> float:
    """Apply drug effect to parameter value.

    Uses Hill equation for dose-response:
        Effect = Efficacy * C^n / (EC50^n + C^n)
    """
    return target.effect_function(baseline_value, concentration_uM)


def apply_drug(digital_twin: ENSGIDigitalTwin,
              drug: DrugProfile,
              dose_mg: float,
              time_hours: float = 2.0) -> Dict[str, float]:
    """Apply drug to digital twin by modifying parameters.

    Args:
        digital_twin: ENSGIDigitalTwin instance
        drug: DrugProfile to apply
        dose_mg: Dose in milligrams
        time_hours: Time since administration (for PK)

    Returns:
        Dict of modified parameters
    """
    # Compute plasma concentration
    concentration = compute_plasma_concentration(dose_mg, drug, time_hours)

    modified_params = {'concentration_uM': concentration}

    # Apply each target effect
    for target in drug.targets:
        param_name = target.parameter

        # Find where this parameter lives
        if hasattr(digital_twin.network.neurons[0].params, param_name):
            # Membrane parameter
            for neuron in digital_twin.network.neurons:
                baseline = getattr(neuron.params, param_name)
                new_value = apply_drug_effect(baseline, concentration, target)
                setattr(neuron.params, param_name, new_value)
                modified_params[param_name] = new_value

        elif hasattr(digital_twin.network.params, param_name):
            # Network parameter
            baseline = getattr(digital_twin.network.params, param_name)
            new_value = apply_drug_effect(baseline, concentration, target)
            setattr(digital_twin.network.params, param_name, new_value)
            modified_params[param_name] = new_value

        elif hasattr(digital_twin.icc.params, param_name):
            # ICC parameter
            baseline = getattr(digital_twin.icc.params, param_name)
            new_value = apply_drug_effect(baseline, concentration, target)
            setattr(digital_twin.icc.params, param_name, new_value)
            modified_params[param_name] = new_value

    # Rebuild network connectivity if coupling changed
    if 'coupling_strength' in modified_params or 'excitatory_weight' in modified_params:
        digital_twin.network.connections = digital_twin.network._build_connections()

    return modified_params


# ═══════════════════════════════════════════════════════════════
# Virtual Drug Trials
# ═══════════════════════════════════════════════════════════════

@dataclass
class TrialResults:
    """Results from virtual drug trial."""
    drug_name: str
    n_patients: int
    doses_tested: List[float]

    # Biomarker outcomes (drug vs placebo)
    drug_biomarkers: List[Dict]  # List of biomarker dicts per patient
    placebo_biomarkers: List[Dict]

    # Statistical analysis
    p_values: Dict[str, float]
    effect_sizes: Dict[str, float]
    responder_rate: float  # Fraction showing >30% improvement

    # Dose-response
    optimal_dose_mg: float
    dose_response_curve: Dict[float, float]  # dose -> mean response


class VirtualDrugTrial:
    """Conduct virtual clinical trial for drug efficacy."""

    def __init__(self,
                 drug: DrugProfile,
                 cohort_size: int = 100,
                 patient_profile: str = 'ibs_c',
                 parameter_variation: float = 0.15):
        """
        Args:
            drug: Drug to test
            cohort_size: Number of virtual patients
            patient_profile: Base IBS profile ('ibs_d', 'ibs_c', 'ibs_m')
            parameter_variation: Inter-patient variability (CV)
        """
        self.drug = drug
        self.cohort_size = cohort_size
        self.patient_profile = patient_profile
        self.parameter_variation = parameter_variation

        print(f"[Virtual Trial] {drug.name} for {patient_profile.upper()}")
        print(f"[Virtual Trial] Cohort size: {cohort_size}")

    def generate_patient_cohort(self) -> List[ENSGIDigitalTwin]:
        """Generate cohort of virtual patients with parameter variation."""
        patients = []

        for i in range(self.cohort_size):
            # Create base twin
            twin = ENSGIDigitalTwin(n_segments=12)
            twin.apply_profile(self.patient_profile)

            # Add inter-patient variability
            for neuron in twin.network.neurons:
                for param_name in ['g_Na', 'g_K', 'g_Ca', 'g_L']:
                    baseline = getattr(neuron.params, param_name)
                    varied = baseline * (1 + np.random.randn() * self.parameter_variation)
                    varied = max(varied, baseline * 0.5)  # Don't go too low
                    setattr(neuron.params, param_name, varied)

            patients.append(twin)

        return patients

    def run_trial(self,
                 doses_mg: Optional[List[float]] = None,
                 simulation_duration: float = 2000.0) -> TrialResults:
        """Run complete virtual trial.

        Args:
            doses_mg: List of doses to test (uses standard if None)
            simulation_duration: Simulation time per patient (ms)

        Returns:
            TrialResults with outcomes and statistics
        """
        if doses_mg is None:
            doses_mg = [self.drug.standard_dose_mg]

        print(f"\n[Trial] Generating {self.cohort_size} virtual patients...")
        patients = self.generate_patient_cohort()

        # Run placebo arm (baseline)
        print(f"[Trial] Running placebo arm...")
        placebo_biomarkers = []
        for i, patient in enumerate(patients):
            patient.run(simulation_duration, dt=0.05, I_stim={3: 10.0}, verbose=False)
            bio = patient.extract_biomarkers()
            placebo_biomarkers.append(bio)
            if (i+1) % 20 == 0:
                print(f"  {i+1}/{self.cohort_size} patients...")

        # Run drug arm (for each dose)
        dose_response = {}

        for dose in doses_mg:
            print(f"\n[Trial] Testing dose: {dose} mg...")
            drug_biomarkers = []

            # Re-generate patients (fresh cohort for each dose)
            patients_drug = self.generate_patient_cohort()

            for i, patient in enumerate(patients_drug):
                # Apply drug
                apply_drug(patient, self.drug, dose_mg=dose, time_hours=2.0)

                # Run simulation
                patient.run(simulation_duration, dt=0.05, I_stim={3: 10.0}, verbose=False)
                bio = patient.extract_biomarkers()
                drug_biomarkers.append(bio)

                if (i+1) % 20 == 0:
                    print(f"  {i+1}/{self.cohort_size} patients...")

            # Compute mean response
            mean_motility_drug = np.mean([b['motility_index'] for b in drug_biomarkers])
            mean_motility_placebo = np.mean([b['motility_index'] for b in placebo_biomarkers])
            dose_response[dose] = mean_motility_drug - mean_motility_placebo

        # Statistical analysis (using standard dose)
        drug_biomarkers_std = []
        patients_std = self.generate_patient_cohort()
        for patient in patients_std:
            apply_drug(patient, self.drug, dose_mg=self.drug.standard_dose_mg)
            patient.run(simulation_duration, dt=0.05, I_stim={3: 10.0}, verbose=False)
            drug_biomarkers_std.append(patient.extract_biomarkers())

        p_values = {}
        effect_sizes = {}

        for key in ['motility_index', 'spike_rate_per_neuron', 'mean_contractile_force']:
            drug_vals = np.array([b[key] for b in drug_biomarkers_std])
            placebo_vals = np.array([b[key] for b in placebo_biomarkers])

            # T-test
            from scipy import stats
            t_stat, p_val = stats.ttest_ind(drug_vals, placebo_vals)
            p_values[key] = p_val

            # Cohen's d effect size
            pooled_std = np.sqrt((np.std(drug_vals)**2 + np.std(placebo_vals)**2) / 2)
            effect_size = (np.mean(drug_vals) - np.mean(placebo_vals)) / pooled_std
            effect_sizes[key] = effect_size

        # Responder rate (>30% improvement in motility)
        improvements = []
        for i in range(len(drug_biomarkers_std)):
            drug_motility = drug_biomarkers_std[i]['motility_index']
            placebo_motility = placebo_biomarkers[i]['motility_index']
            improvement = (drug_motility - placebo_motility) / (placebo_motility + 1e-6)
            improvements.append(improvement)

        responder_rate = np.mean(np.array(improvements) > 0.3)

        # Find optimal dose
        optimal_dose = max(dose_response, key=dose_response.get)

        results = TrialResults(
            drug_name=self.drug.name,
            n_patients=self.cohort_size,
            doses_tested=doses_mg,
            drug_biomarkers=drug_biomarkers_std,
            placebo_biomarkers=placebo_biomarkers,
            p_values=p_values,
            effect_sizes=effect_sizes,
            responder_rate=responder_rate,
            optimal_dose_mg=optimal_dose,
            dose_response_curve=dose_response
        )

        return results

    @staticmethod
    def print_results(results: TrialResults):
        """Print trial results summary."""
        print("\n" + "="*70)
        print(f"VIRTUAL TRIAL RESULTS: {results.drug_name}")
        print("="*70)
        print(f"Cohort Size: {results.n_patients}")
        print(f"Doses Tested: {results.doses_tested}")
        print(f"\nOptimal Dose: {results.optimal_dose_mg} mg")
        print(f"Responder Rate: {results.responder_rate*100:.1f}%")

        print(f"\nStatistical Outcomes:")
        print(f"{'Biomarker':<30} {'p-value':<12} {'Effect Size (d)':<15} {'Significant?':<12}")
        print("-"*70)

        for key in results.p_values.keys():
            p_val = results.p_values[key]
            effect = results.effect_sizes[key]
            sig = "Yes ***" if p_val < 0.001 else ("Yes **" if p_val < 0.01 else ("Yes *" if p_val < 0.05 else "No"))
            print(f"{key:<30} {p_val:<12.4f} {effect:<15.2f} {sig:<12}")

        print("\nDose-Response Curve:")
        for dose, response in results.dose_response_curve.items():
            print(f"  {dose:>6.1f} mg → {response:>+8.2f} change in motility index")

        print("="*70)


# ═══════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════

def demo_drug_library():
    """Demonstrate drug library and virtual trials."""
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  ENS-GI Drug Library — Virtual Trial Demo                ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # Test 1: Apply single drug
    print("[Demo 1] Testing Mexiletine on IBS-C patient...")
    twin = ENSGIDigitalTwin(n_segments=12)
    twin.apply_profile('ibs_c')

    # Baseline
    twin.run(1500, dt=0.05, I_stim={3: 10.0}, verbose=False)
    baseline_bio = twin.extract_biomarkers()
    print(f"  Baseline motility index: {baseline_bio['motility_index']:.2f}")

    # Apply Mexiletine
    twin_drug = ENSGIDigitalTwin(n_segments=12)
    twin_drug.apply_profile('ibs_c')
    modified = apply_drug(twin_drug, DrugLibrary.MEXILETINE, dose_mg=200)
    print(f"  Drug applied: {modified}")

    twin_drug.run(1500, dt=0.05, I_stim={3: 10.0}, verbose=False)
    drug_bio = twin_drug.extract_biomarkers()
    print(f"  Post-drug motility index: {drug_bio['motility_index']:.2f}")
    print(f"  Improvement: {((drug_bio['motility_index'] - baseline_bio['motility_index'])/baseline_bio['motility_index']*100):.1f}%")

    # Test 2: Virtual trial
    print("\n[Demo 2] Running virtual trial (small cohort for demo)...")
    trial = VirtualDrugTrial(
        drug=DrugLibrary.MEXILETINE,
        cohort_size=20,  # Small for demo
        patient_profile='ibs_c'
    )

    results = trial.run_trial(
        doses_mg=[100, 200, 300],
        simulation_duration=1500
    )

    VirtualDrugTrial.print_results(results)

    print("\n✓ Drug library demo complete!")
    print("\nAvailable drugs:")
    for drug in DrugLibrary.get_all_drugs():
        print(f"  • {drug.name} ({drug.drug_class}) - {drug.indication}")


if __name__ == '__main__':
    demo_drug_library()
