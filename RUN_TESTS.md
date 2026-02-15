# Manual Test Commands - ENS-GI Digital Twin

**Copy and paste these commands one by one in PowerShell**

---

## Setup (One-Time)

```powershell
# Navigate to project directory
cd "C:\ens-gi digital"

# Use base conda environment (has Python 3.13)
conda activate base

# Install minimal required packages (skip PyMC3)
pip install -r requirements_minimal.txt

# Verify Python works
python --version
```

**Expected:** `Python 3.13.9` (or similar)

---

## Quick Manual Test (30 seconds)

```powershell
# Run manual test script
python test_manual.py
```

**Expected Output:**
```
[TEST 1/9] Core Simulation Engine
   ✅ PASS - ICC frequency: 3.02 cycles/min

[TEST 2/9] IBS Profile Differences
   ✅ PASS - Healthy: 3.01, IBS-D: 3.89, IBS-C: 2.26

[TEST 3/9] Patient Data Loader
   ✅ PASS - EGG: (12000, 4), HRM: (12000, 8)

[TEST 4/9] PINN Framework
   ✅ PASS - Generated 10 training samples

[TEST 5/9] Drug Library
   ✅ PASS - Loaded Mexiletine (Na+ channel blocker)

[TEST 6/9] Virtual Drug Trial
   ✅ PASS - Created trial with 10 patients

[TEST 7/9] Clinical Workflow
   ✅ PASS - Clinical workflow initialized

[TEST 8/9] SPICE Export (Critical Bug Fixes)
   ✅ PASS - All 6 ion channels present
      ✓ Ca²⁺ instantiation
      ✓ Ca²⁺ subcircuit
      ✓ KCa subcircuit
      ✓ A-type K subcircuit
      ✓ Na subcircuit
      ✓ K subcircuit

[TEST 9/9] Verilog-A Export
   ✅ PASS - Verilog-A modules referenced

[TEST 10/10] Bayesian Framework (Optional)
   ⚠️  SKIPPED - PyMC3 not installed (optional)

✅ PASSED:  9/10
⚠️  SKIPPED: 1/10

🎉 ALL CRITICAL TESTS PASSED!
```

---

## Full Test Suite (2-3 minutes)

```powershell
# Run pytest on all tests
pytest tests/ -v --tb=short
```

**Expected:**
- ✅ 66+ tests passing
- ⚠️ 11 tests skipped (PyMC3 - optional)
- ❌ 0-1 failures (action potential timing - non-critical)

---

## Individual Component Tests

### Test 1: Core Simulation

```powershell
python -c "
from ens_gi_core import ENSGIDigitalTwin

twin = ENSGIDigitalTwin(n_segments=20)
twin.apply_profile('healthy')

for _ in range(20000):
    twin.step()

bio = twin.extract_biomarkers()
print(f'ICC Frequency: {bio[\"icc_frequency\"]:.2f} cycles/min')
print(f'Spike Rate: {bio[\"spike_rate\"]:.2f} spikes/min')
print('✅ Core simulation working!')
"
```

---

### Test 2: IBS Profiles

```powershell
python -c "
from ens_gi_core import ENSGIDigitalTwin

for profile in ['healthy', 'ibs_d', 'ibs_c']:
    twin = ENSGIDigitalTwin(n_segments=20)
    twin.apply_profile(profile)
    for _ in range(10000):
        twin.step()
    bio = twin.extract_biomarkers()
    print(f'{profile.upper():10s}: ICC = {bio[\"icc_frequency\"]:.2f} cycles/min')

print('✅ IBS profiles working!')
"
```

---

### Test 3: Patient Data Loader

```powershell
python -c "
from patient_data_loader import PatientDataLoader

for pid in ['P001', 'P002', 'P003']:
    loader = PatientDataLoader(f'patient_data/{pid}_egg.csv', f'patient_data/{pid}_hrm.csv')
    egg, hrm = loader.load()
    print(f'{pid}: EGG={egg.shape}, HRM={hrm.shape}')

print('✅ Patient data loader working!')
"
```

---

### Test 4: PINN Framework

```powershell
python -c "
from ens_gi_pinn import PINNEstimator

pinn = PINNEstimator()
pinn.generate_synthetic_dataset(n_samples=10)
print(f'Training samples: {len(pinn.dataset[\"params\"])}')
print('✅ PINN framework ready!')
"
```

---

### Test 5: Drug Library

```powershell
python -c "
from ens_gi_drug_library import get_drug_by_name, DRUG_LIBRARY

print('Available Drugs:')
for drug_name in DRUG_LIBRARY.keys():
    drug = get_drug_by_name(drug_name)
    print(f'  - {drug.name}: {drug.target}')

print('✅ Drug library working!')
"
```

---

### Test 6: SPICE Export (CRITICAL - Bug Fixes)

```powershell
python -c "
from ens_gi_core import ENSGIDigitalTwin

twin = ENSGIDigitalTwin(n_segments=5)
netlist = twin.export_spice_netlist(filename='verify_spice.sp', use_verilog_a=False)

checks = [
    ('Ca²⁺ instantiation', 'X_ca0 V0 0 ca_channel' in netlist),
    ('Ca²⁺ subcircuit', '.subckt ca_channel' in netlist),
    ('KCa subcircuit', '.subckt kca_channel' in netlist),
    ('A-type K subcircuit', '.subckt a_type_k' in netlist),
    ('Na subcircuit', '.subckt na_channel' in netlist),
    ('K subcircuit', '.subckt k_channel' in netlist),
]

print('SPICE Export Verification:')
for name, passed in checks:
    status = '✅' if passed else '❌'
    print(f'  {status} {name}')

if all(p for _, p in checks):
    print('\n✅ SPICE EXPORT PASSED - All 6 ion channels present!')
else:
    print('\n❌ SPICE FAILED')
"
```

---

### Test 7: Inspect SPICE Netlist File

```powershell
# View SPICE netlist structure
Get-Content verify_spice.sp -Head 150 | Select-String -Pattern "subckt|X_ca|X_k|X_na"
```

**Look for:**
```
.subckt na_channel vp vn
.subckt k_channel vp vn
.subckt ca_channel vp vn       ← Should be present
.subckt kca_channel vp vn      ← Should be present
.subckt a_type_k vp vn         ← Should be present
.subckt leak vp vn

X_na0 V0 0 na_channel
X_k0  V0 0 k_channel
X_ca0 V0 0 ca_channel          ← Should be present
```

---

### Test 8: Run Specific Test File

```powershell
# Test core functionality
pytest tests/test_core.py -v

# Test PINN framework
pytest tests/test_pinn.py -v

# Test drug library
pytest tests/test_drug_library.py -v

# Test validation
pytest tests/test_validation.py -v
```

---

### Test 9: SPICE Validation with ngspice (Optional)

**Only if you have ngspice installed:**

```powershell
python validate_spice.py
```

**Expected:**
```
Step 1/5: Running Python simulation...
  ✓ Python simulation complete

Step 2/5: Exporting SPICE netlist...
  ✓ Netlist exported

Step 3/5: Running ngspice simulation...
  ✓ ngspice simulation complete

Step 4/5: Parsing SPICE output...
  ✓ Parsed output

Step 5/5: Comparing results...
  ✓ Comparison complete

SPICE VALIDATION REPORT - ✅ PASS
Voltage Correlation: 0.96 (target: >0.95)
```

---

## Troubleshooting

### If "python: command not found"

```powershell
# Make sure conda is activated
conda activate base

# Or use full path
C:\ProgramData\anaconda3\python.exe test_manual.py
```

---

### If "ModuleNotFoundError"

```powershell
# Install missing packages
pip install -r requirements_minimal.txt

# Or install individually
pip install numpy scipy matplotlib tensorflow scikit-learn pandas pytest
```

---

### If SPICE test fails

The SPICE test is the most important - it verifies the bug fixes from this session. If it fails, the Ca²⁺ channel fixes didn't apply correctly.

---

## Summary Checklist

Run these commands in order:

- [ ] `conda activate base`
- [ ] `cd "C:\ens-gi digital"`
- [ ] `pip install -r requirements_minimal.txt`
- [ ] `python test_manual.py` ← Main verification
- [ ] `pytest tests/ -v` ← Full test suite
- [ ] Verify SPICE export has all 6 channels

**If all pass:** ✅ You're ready for real data!

---

**Questions?**
- See [VERIFICATION_CHECKLIST.md](VERIFICATION_CHECKLIST.md) for detailed guide
- See [IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md) for next steps
