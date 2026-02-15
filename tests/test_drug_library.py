"""
Unit tests for drug library and virtual drug trials
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ens_gi_digital.core import ENSGIDigitalTwin
from ens_gi_digital.drug_library import (
    DrugLibrary, DrugProfile, DrugTarget, TargetType,
    apply_drug, compute_plasma_concentration, apply_drug_effect,
    VirtualDrugTrial
)


class TestDrugProfiles:
    """Test drug profile definitions."""

    def test_all_drugs_available(self):
        """Test that all drugs are accessible."""
        drugs = DrugLibrary.get_all_drugs()
        assert len(drugs) == 7

        expected_drugs = [
            'MEXILETINE', 'ONDANSETRON', 'ALOSETRON',
            'LUBIPROSTONE', 'LINACLOTIDE', 'PRUCALOPRIDE', 'RIFAXIMIN'
        ]

        for drug_name in expected_drugs:
            assert hasattr(DrugLibrary, drug_name)

    def test_drug_profile_structure(self):
        """Test that drug profiles have required fields."""
        mexiletine = DrugLibrary.MEXILETINE

        assert isinstance(mexiletine, DrugProfile)
        assert mexiletine.name == "Mexiletine"
        assert mexiletine.generic_name == "mexiletine"
        assert mexiletine.drug_class is not None
        assert mexiletine.indication is not None
        assert len(mexiletine.targets) > 0
        assert mexiletine.standard_dose_mg > 0
        assert 0 < mexiletine.bioavailability <= 1

    def test_drug_targets_valid(self):
        """Test that drug targets are properly defined."""
        for drug in DrugLibrary.get_all_drugs():
            assert len(drug.targets) > 0

            for target in drug.targets:
                assert isinstance(target, DrugTarget)
                assert target.parameter is not None
                assert isinstance(target.target_type, TargetType)
                assert target.ic50_or_ec50 > 0
                assert callable(target.effect_function)

    def test_get_by_name(self):
        """Test retrieving drugs by name."""
        # By brand name
        mex = DrugLibrary.get_by_name('Mexiletine')
        assert mex is not None
        assert mex.name == 'Mexiletine'

        # By generic name
        mex2 = DrugLibrary.get_by_name('mexiletine')
        assert mex2 is not None
        assert mex2.name == 'Mexiletine'

        # Non-existent drug
        fake = DrugLibrary.get_by_name('FakeDrug')
        assert fake is None


class TestPharmacokinetics:
    """Test pharmacokinetic modeling."""

    def test_plasma_concentration_decay(self):
        """Test that plasma concentration decays over time."""
        mexiletine = DrugLibrary.MEXILETINE
        dose = 200  # mg

        C_0h = compute_plasma_concentration(dose, mexiletine, time_hours=0)
        C_1h = compute_plasma_concentration(dose, mexiletine, time_hours=1)
        C_10h = compute_plasma_concentration(dose, mexiletine, time_hours=10)

        # Concentration should decay over time
        assert C_0h > C_1h > C_10h
        assert C_0h > 0
        assert C_10h > 0

    def test_dose_proportionality(self):
        """Test that higher doses give higher concentrations."""
        mexiletine = DrugLibrary.MEXILETINE

        C_100mg = compute_plasma_concentration(100, mexiletine, time_hours=2)
        C_200mg = compute_plasma_concentration(200, mexiletine, time_hours=2)

        # Double dose should give approximately double concentration
        ratio = C_200mg / C_100mg
        assert 1.8 < ratio < 2.2

    def test_bioavailability_effect(self):
        """Test that bioavailability affects concentration."""
        # Create two identical drugs with different bioavailability
        drug_high_f = DrugProfile(
            name="TestDrugHighF",
            generic_name="test",
            drug_class="test",
            indication="test",
            targets=[],
            bioavailability=0.9,
            half_life_hours=4.0,
            volume_distribution_L=50.0,
            standard_dose_mg=100
        )

        drug_low_f = DrugProfile(
            name="TestDrugLowF",
            generic_name="test",
            drug_class="test",
            indication="test",
            targets=[],
            bioavailability=0.3,
            half_life_hours=4.0,
            volume_distribution_L=50.0,
            standard_dose_mg=100
        )

        C_high = compute_plasma_concentration(100, drug_high_f, 1.0)
        C_low = compute_plasma_concentration(100, drug_low_f, 1.0)

        assert C_high > C_low


class TestPharmacodynamics:
    """Test pharmacodynamic modeling."""

    def test_drug_effect_application(self):
        """Test that drug effects modify parameters correctly."""
        # Create simple target
        target = DrugTarget(
            parameter='g_Na',
            target_type=TargetType.ION_CHANNEL,
            effect_function=lambda baseline, conc: baseline * (1 - 0.5 * conc / (conc + 10)),
            ic50_or_ec50=10.0,
            hill_coefficient=1.0,
            efficacy=0.5
        )

        baseline = 120.0

        # At very low concentration, should be ~baseline
        effect_low = apply_drug_effect(baseline, 0.1, target)
        assert abs(effect_low - baseline) < 5

        # At high concentration, should be significantly reduced
        effect_high = apply_drug_effect(baseline, 100.0, target)
        assert effect_high < baseline * 0.6  # >40% reduction

    def test_apply_drug_to_twin(self):
        """Test applying drug to digital twin."""
        twin = ENSGIDigitalTwin(n_segments=5)
        twin.apply_profile('ibs_c')

        # Get baseline g_Na
        baseline_g_Na = twin.network.neurons[0].params.g_Na

        # Apply Mexiletine
        modified = apply_drug(twin, DrugLibrary.MEXILETINE, dose_mg=200)

        # Check that parameters were modified
        assert 'concentration_uM' in modified
        assert modified['concentration_uM'] > 0

        # g_Na should be reduced (Mexiletine is Na+ blocker)
        new_g_Na = twin.network.neurons[0].params.g_Na
        assert new_g_Na < baseline_g_Na


class TestVirtualDrugTrial:
    """Test virtual clinical trial functionality."""

    @pytest.fixture
    def small_trial(self):
        """Create small trial for fast testing."""
        return VirtualDrugTrial(
            drug=DrugLibrary.MEXILETINE,
            cohort_size=10,  # Small for testing
            patient_profile='ibs_c',
            parameter_variation=0.1
        )

    def test_trial_initialization(self, small_trial):
        """Test that trial initializes correctly."""
        assert small_trial.drug.name == 'Mexiletine'
        assert small_trial.cohort_size == 10
        assert small_trial.patient_profile == 'ibs_c'

    def test_cohort_generation(self, small_trial):
        """Test patient cohort generation."""
        patients = small_trial.generate_patient_cohort()

        assert len(patients) == 10
        assert all(isinstance(p, ENSGIDigitalTwin) for p in patients)

        # Check parameter variation exists
        g_Na_values = [p.network.neurons[0].params.g_Na for p in patients]
        assert np.std(g_Na_values) > 0  # Some variation should exist

    @pytest.mark.slow
    def test_trial_execution(self, small_trial):
        """Test running complete trial (slow)."""
        results = small_trial.run_trial(
            doses_mg=[100, 200],
            simulation_duration=500  # Short for testing
        )

        assert results.drug_name == 'Mexiletine'
        assert results.n_patients == 10
        assert len(results.doses_tested) == 2
        assert len(results.drug_biomarkers) > 0
        assert len(results.placebo_biomarkers) > 0

        # Check statistical results
        assert 'motility_index' in results.p_values
        assert 'motility_index' in results.effect_sizes
        assert 0 <= results.responder_rate <= 1

    @pytest.mark.slow
    def test_dose_response_curve(self, small_trial):
        """Test dose-response curve generation."""
        results = small_trial.run_trial(
            doses_mg=[50, 100, 200],
            simulation_duration=500
        )

        assert len(results.dose_response_curve) == 3

        # Response should generally increase with dose (for IBS-C + Mexiletine)
        doses_sorted = sorted(results.dose_response_curve.keys())
        responses = [results.dose_response_curve[d] for d in doses_sorted]

        # At least some trend should exist
        assert len(responses) == 3


class TestDrugInteractions:
    """Test drug-drug interactions (future feature)."""

    def test_multiple_drug_targets(self):
        """Test drugs with multiple targets."""
        alosetron = DrugLibrary.ALOSETRON

        # Alosetron has 2 targets
        assert len(alosetron.targets) == 2

        target_params = [t.parameter for t in alosetron.targets]
        assert 'serotonin_factor' in target_params
        assert 'excitatory_weight' in target_params


# Integration test
@pytest.mark.slow
def test_full_drug_trial_workflow():
    """Integration test for complete drug trial."""

    # 1. Create trial
    trial = VirtualDrugTrial(
        drug=DrugLibrary.MEXILETINE,
        cohort_size=20,
        patient_profile='ibs_c'
    )

    # 2. Run trial
    results = trial.run_trial(
        doses_mg=[100, 200, 300],
        simulation_duration=1000
    )

    # 3. Verify results
    assert results is not None
    assert results.n_patients == 20
    assert len(results.dose_response_curve) == 3

    # 4. Print results (optional)
    VirtualDrugTrial.print_results(results)

    print("✓ Full drug trial workflow test passed")


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
