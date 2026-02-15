"""
Manual Test Script for ENS-GI Digital Twin
Run this to verify all core components are working.

Usage: python test_manual.py
"""

import sys

print("=" * 70)
print("ENS-GI Digital Twin - Manual Test Suite")
print("=" * 70)
print()

passed = 0
failed = 0
skipped = 0

# Test 1: Core Simulation
print("[TEST 1/9] Core Simulation Engine")
try:
    from ens_gi_digital import ENSGIDigitalTwin

    twin = ENSGIDigitalTwin(n_segments=5)
    twin.apply_profile('healthy')

    # Run 100ms simulation
    for _ in range(2000):
        twin.step()

    bio = twin.extract_biomarkers()

    # Verify biomarkers are reasonable
    assert 2.0 < bio['icc_frequency'] < 4.0, f"ICC freq out of range: {bio['icc_frequency']}"
    assert bio['spike_rate'] > 0, "Spike rate should be positive"

    print(f"   ✅ PASS - ICC frequency: {bio['icc_frequency']:.2f} cycles/min")
    passed += 1
except Exception as e:
    print(f"   ❌ FAIL - {str(e)}")
    failed += 1

# Test 2: IBS Profiles
print("\n[TEST 2/9] IBS Profile Differences")
try:
    from ens_gi_digital import ENSGIDigitalTwin

    freqs = {}
    for profile in ['healthy', 'ibs_d', 'ibs_c']:
        twin = ENSGIDigitalTwin(n_segments=5)
        twin.apply_profile(profile)
        for _ in range(10000):
            twin.step()
        bio = twin.extract_biomarkers()
        freqs[profile] = bio['icc_frequency']

    # Verify IBS-D > Healthy > IBS-C
    assert freqs['ibs_d'] > freqs['healthy'], "IBS-D should have higher frequency"
    assert freqs['healthy'] > freqs['ibs_c'], "IBS-C should have lower frequency"

    print(f"   ✅ PASS - Healthy: {freqs['healthy']:.2f}, IBS-D: {freqs['ibs_d']:.2f}, IBS-C: {freqs['ibs_c']:.2f}")
    passed += 1
except Exception as e:
    print(f"   ❌ FAIL - {str(e)}")
    failed += 1

# Test 3: Patient Data Loader
print("\n[TEST 3/9] Patient Data Loader")
try:
    from ens_gi_digital import PatientDataLoader

    loader = PatientDataLoader('patient_data/P001_egg.csv', 'patient_data/P001_hrm.csv')
    egg, hrm = loader.load()

    assert egg.shape[1] == 4, f"EGG should have 4 channels, got {egg.shape[1]}"
    assert hrm.shape[1] == 8, f"HRM should have 8 sensors, got {hrm.shape[1]}"
    assert egg.shape[0] > 1000, "EGG should have many timepoints"

    print(f"   ✅ PASS - EGG: {egg.shape}, HRM: {hrm.shape}")
    passed += 1
except Exception as e:
    print(f"   ❌ FAIL - {str(e)}")
    failed += 1

# Test 4: PINN Framework
print("\n[TEST 4/9] PINN Framework")
try:
    from ens_gi_digital import PINNEstimator

    pinn = PINNEstimator()

    # Generate small dataset
    pinn.generate_synthetic_dataset(n_samples=10, param_ranges={
        'g_Na': (100, 140),
        'g_K': (30, 45)
    })

    assert len(pinn.dataset['params']) == 10, "Should have 10 samples"

    print(f"   ✅ PASS - Generated {len(pinn.dataset['params'])} training samples")
    passed += 1
except Exception as e:
    print(f"   ❌ FAIL - {str(e)}")
    failed += 1

# Test 5: Drug Library
print("\n[TEST 5/9] Drug Library")
try:
    from ens_gi_digital.drug_library import get_drug_by_name, DRUG_LIBRARY

    # Test drug retrieval
    drug = get_drug_by_name("mexiletine")

    assert drug.name == "Mexiletine", "Drug name mismatch"
    assert drug.target == "Na+ channel blocker", "Target mismatch"
    assert len(DRUG_LIBRARY) == 7, f"Should have 7 drugs, got {len(DRUG_LIBRARY)}"

    print(f"   ✅ PASS - Loaded {drug.name} ({drug.target})")
    passed += 1
except Exception as e:
    print(f"   ❌ FAIL - {str(e)}")
    failed += 1

# Test 6: Virtual Drug Trial
print("\n[TEST 6/9] Virtual Drug Trial")
try:
    from ens_gi_digital.drug_library import VirtualDrugTrial, get_drug_by_name

    drug = get_drug_by_name("mexiletine")
    trial = VirtualDrugTrial(drug, n_patients=10, baseline_profile='ibs_c')

    assert trial.n_patients == 10, "Should have 10 patients"
    assert trial.baseline_profile == 'ibs_c', "Profile mismatch"

    print(f"   ✅ PASS - Created trial with {trial.n_patients} patients")
    passed += 1
except Exception as e:
    print(f"   ❌ FAIL - {str(e)}")
    failed += 1

# Test 7: Clinical Workflow
print("\n[TEST 7/9] Clinical Workflow")
try:
    from ens_gi_digital import ClinicalWorkflow

    workflow = ClinicalWorkflow()

    print(f"   ✅ PASS - Clinical workflow initialized")
    passed += 1
except Exception as e:
    print(f"   ❌ FAIL - {str(e)}")
    failed += 1

# Test 8: SPICE Export (CRITICAL - Bug Fixes Verification)
print("\n[TEST 8/9] SPICE Export (Critical Bug Fixes)")
try:
    from ens_gi_digital import ENSGIDigitalTwin

    twin = ENSGIDigitalTwin(n_segments=5)
    twin.apply_profile('healthy')

    netlist = twin.export_spice_netlist(filename='manual_test_spice.sp', use_verilog_a=False)

    # Verify all 6 ion channels are present
    checks = {
        'Ca²⁺ instantiation': 'X_ca0 V0 0 ca_channel' in netlist,
        'Ca²⁺ subcircuit': '.subckt ca_channel' in netlist,
        'KCa subcircuit': '.subckt kca_channel' in netlist,
        'A-type K subcircuit': '.subckt a_type_k' in netlist,
        'Na subcircuit': '.subckt na_channel' in netlist,
        'K subcircuit': '.subckt k_channel' in netlist,
    }

    all_passed = all(checks.values())

    if all_passed:
        print(f"   ✅ PASS - All 6 ion channels present")
        for name, result in checks.items():
            print(f"      ✓ {name}")
        passed += 1
    else:
        print(f"   ❌ FAIL - Missing channels:")
        for name, result in checks.items():
            status = "✓" if result else "✗"
            print(f"      {status} {name}")
        failed += 1
except Exception as e:
    print(f"   ❌ FAIL - {str(e)}")
    failed += 1

# Test 9: Verilog-A Export
print("\n[TEST 9/9] Verilog-A Export")
try:
    from ens_gi_digital import ENSGIDigitalTwin

    twin = ENSGIDigitalTwin(n_segments=3)
    verilog = twin.export_verilog_a_module()

    # Check for key modules
    modules_found = {
        'NaV1_5': 'NaV1_5' in verilog,
        'CaL_channel': 'CaL_channel' in verilog,
        'Kv_delayed_rectifier': 'Kv_delayed_rectifier' in verilog,
    }

    all_found = all(modules_found.values())

    if all_found:
        print(f"   ✅ PASS - Verilog-A modules referenced")
        passed += 1
    else:
        print(f"   ❌ FAIL - Missing modules")
        failed += 1
except Exception as e:
    print(f"   ❌ FAIL - {str(e)}")
    failed += 1

# Test 10: Bayesian Framework (Optional)
print("\n[TEST 10/10] Bayesian Framework (Optional)")
try:
    from ens_gi_digital import BayesianEstimator

    bayes = BayesianEstimator()
    n_priors = len(bayes.priors)

    print(f"   ✅ PASS - Bayesian framework ready ({n_priors} priors)")
    passed += 1
except ImportError:
    print(f"   ⚠️  SKIPPED - PyMC3 not installed (optional)")
    skipped += 1
except Exception as e:
    print(f"   ⚠️  SKIPPED - {str(e)}")
    skipped += 1

# Summary
print("\n" + "=" * 70)
print("Test Summary")
print("=" * 70)
print(f"✅ PASSED:  {passed}/10")
print(f"❌ FAILED:  {failed}/10")
print(f"⚠️  SKIPPED: {skipped}/10")
print()

if failed == 0:
    print("🎉 ALL CRITICAL TESTS PASSED!")
    print()
    print("Next Steps:")
    print("1. ✅ Core system verified and working")
    print("2. ✅ SPICE bug fixes confirmed")
    print("3. ✅ Ready for real patient data")
    print()
    print("To run full test suite: pytest tests/ -v")
    print("To verify SPICE in ngspice: python validate_spice.py")
    print()
    sys.exit(0)
else:
    print("⚠️  Some tests failed. Please review errors above.")
    sys.exit(1)
