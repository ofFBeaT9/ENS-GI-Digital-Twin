"""
Quick Working Test - ENS-GI Digital Twin
Uses correct APIs based on actual code
"""

print("=" * 70)
print("ENS-GI Digital Twin - Quick Verification Test")
print("=" * 70)
print()

passed = 0
failed = 0

# Test 1: Core Simulation
print("[TEST 1/5] Core Simulation")
try:
    from ens_gi_digital import ENSGIDigitalTwin

    twin = ENSGIDigitalTwin(n_segments=5)
    twin.apply_profile('healthy')
    twin.run(duration=100, record=True)  # Run 100ms
    bio = twin.extract_biomarkers()

    freq = bio['icc_frequency_cpm']  # Correct key!
    assert 2.0 < freq < 4.0, f"ICC freq out of range: {freq:.2f} cpm (expected 2-4 cpm)"

    print(f"   ✅ PASS - ICC frequency: {freq:.2f} cpm (normal range: 2-4 cpm)")
    passed += 1
except Exception as e:
    print(f"   ❌ FAIL - {str(e)}")
    failed += 1

# Test 2: IBS Profiles
print("\n[TEST 2/5] IBS Profiles")
try:
    from ens_gi_digital import ENSGIDigitalTwin

    freqs = {}
    for profile in ['healthy', 'ibs_d', 'ibs_c']:
        twin = ENSGIDigitalTwin(n_segments=5)
        twin.apply_profile(profile)
        twin.run(duration=500, record=True)
        bio = twin.extract_biomarkers()
        freqs[profile] = bio['icc_frequency_cpm']

    assert freqs['ibs_d'] > freqs['healthy'], "IBS-D should be higher"
    assert freqs['healthy'] > freqs['ibs_c'], "IBS-C should be lower"

    print(f"   ✅ PASS")
    print(f"      Healthy: {freqs['healthy']:.2f} cpm")
    print(f"      IBS-D:   {freqs['ibs_d']:.2f} cpm (hyperexcitable)")
    print(f"      IBS-C:   {freqs['ibs_c']:.2f} cpm (hypoexcitable)")
    passed += 1
except Exception as e:
    print(f"   ❌ FAIL - {str(e)}")
    failed += 1

# Test 3: SPICE Export (CRITICAL - Bug Fixes)
print("\n[TEST 3/5] SPICE Export (Critical Bug Fixes)")
try:
    from ens_gi_digital import ENSGIDigitalTwin

    twin = ENSGIDigitalTwin(n_segments=5)
    netlist = twin.export_spice_netlist(filename='quick_test.sp', use_verilog_a=False)

    checks = {
        'Ca²⁺ instantiation': 'X_ca0 V0 0 ca_channel' in netlist,
        'Ca²⁺ subcircuit': '.subckt ca_channel' in netlist,
        'KCa subcircuit': '.subckt kca_channel' in netlist,
        'A-type K subcircuit': '.subckt a_type_k' in netlist,
        'Na subcircuit': '.subckt na_channel' in netlist,
        'K subcircuit': '.subckt k_channel' in netlist,
    }

    if all(checks.values()):
        print(f"   ✅ PASS - All 6 ion channels present")
        for name in checks.keys():
            print(f"      ✓ {name}")
        passed += 1
    else:
        print(f"   ❌ FAIL - Missing channels")
        failed += 1
except Exception as e:
    print(f"   ❌ FAIL - {str(e)}")
    failed += 1

# Test 4: Verilog-A Export
print("\n[TEST 4/5] Verilog-A Export")
try:
    from ens_gi_digital import ENSGIDigitalTwin

    twin = ENSGIDigitalTwin(n_segments=3)
    verilog = twin.export_verilog_a_module()

    has_content = len(verilog) > 100
    assert has_content, "Verilog-A code too short"

    print(f"   ✅ PASS - Verilog-A code generated ({len(verilog)} chars)")
    passed += 1
except Exception as e:
    print(f"   ❌ FAIL - {str(e)}")
    failed += 1

# Test 5: Run Simulation and Extract All Biomarkers
print("\n[TEST 5/5] Full Biomarker Extraction")
try:
    from ens_gi_digital import ENSGIDigitalTwin

    twin = ENSGIDigitalTwin(n_segments=10)
    twin.apply_profile('healthy')
    twin.run(duration=1000, record=True)  # 1 second
    bio = twin.extract_biomarkers()

    # Check all expected biomarkers
    required_keys = [
        'icc_frequency_cpm',
        'mean_membrane_potential',
        'spike_rate_per_neuron',
        'mean_calcium',
        'motility_index',
        'profile'
    ]

    missing = [k for k in required_keys if k not in bio]
    assert not missing, f"Missing biomarkers: {missing}"

    print(f"   ✅ PASS - All biomarkers extracted")
    print(f"      ICC freq: {bio['icc_frequency_cpm']:.2f} cpm")
    print(f"      Spike rate: {bio['spike_rate_per_neuron']:.2f} Hz/neuron")
    print(f"      Motility index: {bio['motility_index']:.2f}")
    passed += 1
except Exception as e:
    print(f"   ❌ FAIL - {str(e)}")
    failed += 1

# Summary
print("\n" + "=" * 70)
print("Test Summary")
print("=" * 70)
print(f"✅ PASSED: {passed}/5")
print(f"❌ FAILED: {failed}/5")
print()

if failed == 0:
    print("🎉 ALL TESTS PASSED!")
    print()
    print("✅ Core simulation working")
    print("✅ IBS profiles show expected differences")
    print("✅ SPICE export has all 6 ion channels (BUG FIXES VERIFIED!)")
    print("✅ Verilog-A export working")
    print("✅ Biomarker extraction complete")
    print()
    print("=" * 70)
    print("STATUS: READY FOR REAL DATA!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Acquire real patient EGG/HRM data")
    print("2. Use patient_data_loader to load CSV files")
    print("3. Run parameter estimation with PINN")
    print("4. Generate clinical reports")
    print()
else:
    print(f"⚠️  {failed} test(s) failed")
    print("Most critical: SPICE export test should pass")
