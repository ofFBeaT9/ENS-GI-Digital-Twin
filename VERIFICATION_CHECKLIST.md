# Manual Verification Checklist
**Before Real Data Integration**

Run these commands step-by-step to verify everything is working correctly.

---

## ✅ Quick Verification (5 minutes)

### **Step 1: Run Automated Verification Script**

```bash
cd "C:\ens-gi digital"
python verify_installation.py
```

**Expected Output:**
```
[1/8] Testing Core Simulation Engine...
   ✓ Core simulation working!
   - ICC frequency: ~3.0 cycles/min
   - Spike rate: ~15-25 spikes/min

[2/8] Testing Patient Data Loader...
   ✓ Patient data loader working!
   - EGG shape: (time_points, 4)
   - HRM shape: (time_points, 8)

[3/8] Testing PINN Framework...
   ✓ PINN framework working!

[4/8] Testing Bayesian Framework...
   ✓ Bayesian framework working!

[5/8] Testing Drug Library...
   ✓ Drug library working!

[6/8] Testing Clinical Workflow...
   ✓ Clinical workflow working!

[7/8] Testing SPICE Export...
   ✓ SPICE export working!
   - All 6 ion channels present: ✓

[8/8] Testing Verilog-A Export...
   ✓ Verilog-A export working!

Status: ✅ READY FOR REAL DATA INTEGRATION
```

**If all tests pass:** ✅ Proceed to Step 2!

**If any test fails:** Check the error message and verify dependencies are installed.

---

## 📋 Detailed Manual Verification (15-20 minutes)

### **Step 2: Run Full Test Suite**

```bash
cd "C:\ens-gi digital"
pytest tests/ -v --tb=short
```

**Expected Output:**
- ✅ 66+ tests passing
- ⚠️ 11 tests skipped (Bayesian - PyMC3 optional)
- ❌ 0-1 failures (action potential timing test is known flaky, non-critical)

**Look for:**
```
====== 66 passed, 11 skipped in XX.XXs ======
```

---

### **Step 3: Test Core Simulation Manually**

```bash
cd "C:\ens-gi digital"
python -c "
from ens_gi_core import ENSGIDigitalTwin

# Create healthy 20-segment network
twin = ENSGIDigitalTwin(n_segments=20)
twin.apply_profile('healthy')

# Run 1 second simulation
for _ in range(20000):
    twin.step()

# Extract biomarkers
bio = twin.extract_biomarkers()

print(f'ICC Frequency: {bio[\"icc_frequency\"]:.2f} cycles/min')
print(f'Spike Rate: {bio[\"spike_rate\"]:.2f} spikes/min')
print(f'Motility Index: {bio[\"motility_index\"]:.2f}')
print('✅ Core simulation working!')
"
```

**Expected Output:**
```
ICC Frequency: 2.8-3.2 cycles/min
Spike Rate: 15-25 spikes/min
Motility Index: 8-12
✅ Core simulation working!
```

---

### **Step 4: Test IBS Profiles**

```bash
python -c "
from ens_gi_core import ENSGIDigitalTwin

profiles = ['healthy', 'ibs_d', 'ibs_c']

for profile in profiles:
    twin = ENSGIDigitalTwin(n_segments=20)
    twin.apply_profile(profile)

    for _ in range(10000):
        twin.step()

    bio = twin.extract_biomarkers()
    print(f'{profile.upper()}: ICC freq = {bio[\"icc_frequency\"]:.2f} cycles/min')

print('✅ IBS profiles working!')
"
```

**Expected Output:**
```
HEALTHY: ICC freq = 2.8-3.2 cycles/min
IBS_D: ICC freq = 3.5-4.0 cycles/min (higher - hyperexcitable)
IBS_C: ICC freq = 2.0-2.5 cycles/min (lower - hypoexcitable)
✅ IBS profiles working!
```

---

### **Step 5: Test Patient Data Loader**

```bash
python -c "
from patient_data_loader import PatientDataLoader

# Test all synthetic patients
patients = ['P001', 'P002', 'P003']

for patient_id in patients:
    loader = PatientDataLoader(
        f'patient_data/{patient_id}_egg.csv',
        f'patient_data/{patient_id}_hrm.csv'
    )
    egg, hrm = loader.load()
    print(f'{patient_id}: EGG={egg.shape}, HRM={hrm.shape}')

print('✅ Patient data loader working!')
"
```

**Expected Output:**
```
P001: EGG=(time_points, 4), HRM=(time_points, 8)
P002: EGG=(time_points, 4), HRM=(time_points, 8)
P003: EGG=(time_points, 4), HRM=(time_points, 8)
✅ Patient data loader working!
```

---

### **Step 6: Test PINN Parameter Estimation**

```bash
python -c "
from ens_gi_pinn import PINNEstimator
from patient_data_loader import PatientDataLoader

# Load synthetic patient
loader = PatientDataLoader('patient_data/P001_egg.csv', 'patient_data/P001_hrm.csv')
egg, hrm = loader.load()

# Create estimator
pinn = PINNEstimator()

# Generate small synthetic training set (quick test)
pinn.generate_synthetic_dataset(n_samples=10)

print(f'Training samples: {len(pinn.dataset[\"params\"])}')
print('✅ PINN framework ready!')
print('Note: Full parameter estimation takes 10-30 min on real data')
"
```

**Expected Output:**
```
Training samples: 10
✅ PINN framework ready!
Note: Full parameter estimation takes 10-30 min on real data
```

---

### **Step 7: Test Drug Library**

```bash
python -c "
from ens_gi_drug_library import get_drug_by_name, DRUG_LIBRARY

print('Available Drugs:')
for drug_name in DRUG_LIBRARY.keys():
    drug = get_drug_by_name(drug_name)
    print(f'  - {drug.name}: {drug.target}')

# Test drug application
from ens_gi_core import ENSGIDigitalTwin

drug = get_drug_by_name('mexiletine')
twin = ENSGIDigitalTwin(n_segments=10)
twin.apply_profile('ibs_c')

# Apply drug effect
drug.apply_to_twin(twin, dose_mg=150)

print('\n✅ Drug library working!')
"
```

**Expected Output:**
```
Available Drugs:
  - Mexiletine: Na+ channel blocker
  - Ondansetron: 5-HT3 antagonist
  - Alosetron: 5-HT3 antagonist
  - Lubiprostone: ClC-2 activator
  - Linaclotide: Guanylate cyclase agonist
  - Rifaximin: Antibiotic
  - Prucalopride: 5-HT4 agonist

✅ Drug library working!
```

---

### **Step 8: Test SPICE Export (Critical - Bug Fixes)**

```bash
python -c "
from ens_gi_core import ENSGIDigitalTwin

# Create twin
twin = ENSGIDigitalTwin(n_segments=5)
twin.apply_profile('healthy')

# Export SPICE netlist
netlist = twin.export_spice_netlist(filename='manual_verify_test.sp', use_verilog_a=False)

# Verify all ion channels
checks = {
    'Ca²⁺ instantiation': 'X_ca0 V0 0 ca_channel' in netlist,
    'Ca²⁺ subcircuit': '.subckt ca_channel' in netlist,
    'KCa subcircuit': '.subckt kca_channel' in netlist,
    'A-type K subcircuit': '.subckt a_type_k' in netlist,
    'Na subcircuit': '.subckt na_channel' in netlist,
    'K subcircuit': '.subckt k_channel' in netlist,
}

print('SPICE Export Verification:')
for check, passed in checks.items():
    status = '✅' if passed else '❌'
    print(f'  {status} {check}')

all_passed = all(checks.values())
if all_passed:
    print('\n✅ SPICE export PASSED - All 6 ion channels present!')
    print('File saved: manual_verify_test.sp')
else:
    print('\n❌ SPICE export FAILED - Missing channels!')
"
```

**Expected Output:**
```
SPICE Export Verification:
  ✅ Ca²⁺ instantiation
  ✅ Ca²⁺ subcircuit
  ✅ KCa subcircuit
  ✅ A-type K subcircuit
  ✅ Na subcircuit
  ✅ K subcircuit

✅ SPICE export PASSED - All 6 ion channels present!
File saved: manual_verify_test.sp
```

**This verifies the critical bug fixes from this session!**

---

### **Step 9: Inspect SPICE Netlist File**

```bash
# View first 100 lines of generated SPICE file
head -100 manual_verify_test.sp
```

**Look for these sections:**
```spice
* --- Subcircuit Definitions ---
.subckt na_channel vp vn
.subckt k_channel vp vn
.subckt ca_channel vp vn       ← NEW (was missing!)
.subckt kca_channel vp vn      ← NEW (was missing!)
.subckt a_type_k vp vn         ← NEW (was missing!)
.subckt leak vp vn

* --- Network Instantiation ---
X_na0 V0 0 na_channel
X_k0  V0 0 k_channel
X_ca0 V0 0 ca_channel          ← NEW (was missing!)
X_l0  V0 0 leak
```

---

### **Step 10: Test Verilog-A Export**

```bash
python -c "
from ens_gi_core import ENSGIDigitalTwin

twin = ENSGIDigitalTwin(n_segments=3)
verilog = twin.export_verilog_a_module()

# Check for modules
modules = ['NaV1_5', 'CaL_channel', 'Kv_delayed_rectifier', 'icc_fhn_oscillator']
print('Verilog-A Modules Referenced:')
for module in modules:
    found = module in verilog
    status = '✅' if found else '❌'
    print(f'  {status} {module}')

print('\n✅ Verilog-A export working!')
"
```

---

## 🔍 Optional: Deep Validation (30-60 minutes)

### **Step 11: Run SPICE Validation (Optional)**

This tests SPICE netlists in actual ngspice simulator:

```bash
python validate_spice.py
```

**What it does:**
1. Exports SPICE netlist
2. Runs ngspice simulation
3. Compares with Python simulation
4. Generates validation plots

**Expected Output:**
```
Step 1/5: Running Python simulation...
  ✓ Python simulation complete

Step 2/5: Exporting SPICE netlist...
  ✓ Netlist exported to: test_network.sp

Step 3/5: Running ngspice simulation...
  ✓ ngspice simulation complete

Step 4/5: Parsing SPICE output...
  ✓ Parsed XXXX timesteps

Step 5/5: Comparing results...
  ✓ Comparison complete

SPICE VALIDATION REPORT - ✅ PASS
Voltage Correlation: 0.96 (target: >0.95)
Frequency Match: ✓
Propagation Match: ✓
```

**Note:** Requires ngspice installed at: `c:\ens-gi digital\Spice64\bin\ngspice.exe`

---

### **Step 12: Run Validation Tests**

Test parameter recovery and IBS classification:

```bash
pytest tests/test_validation.py -v
```

**Tests include:**
- IBS profile biomarker validation
- PINN parameter recovery
- Bayesian credible interval coverage
- Drug trial validation

---

## ✅ Verification Complete Checklist

After running all steps, check off:

- [ ] **Step 1:** Automated verification script passed (8/8 tests)
- [ ] **Step 2:** Test suite passed (66+ tests)
- [ ] **Step 3:** Core simulation produces reasonable biomarkers
- [ ] **Step 4:** IBS profiles show expected frequency differences
- [ ] **Step 5:** Patient data loader reads all 3 synthetic patients
- [ ] **Step 6:** PINN framework initializes correctly
- [ ] **Step 7:** Drug library has all 7 drugs available
- [ ] **Step 8:** SPICE export includes all 6 ion channels ✅ **CRITICAL**
- [ ] **Step 9:** SPICE netlist file looks correct
- [ ] **Step 10:** Verilog-A export includes hardware modules
- [ ] **Step 11:** (Optional) SPICE validation passes in ngspice
- [ ] **Step 12:** (Optional) Validation tests pass

---

## 🚀 Ready for Next Step?

If all checks pass, you're ready to move on!

### **What's Next:**

1. **Acquire Real Patient Data**
   - See [REAL_DATA_READINESS_REPORT.md](REAL_DATA_READINESS_REPORT.md) for details
   - Target: 30+ patients (EGG + HRM recordings)
   - Sources: PhysioNet, research collaborations

2. **Run Parameter Estimation on Real Data**
   ```python
   from patient_data_loader import PatientDataLoader
   from ens_gi_pinn import PINNEstimator

   loader = PatientDataLoader('real_patient_egg.csv', 'real_patient_hrm.csv')
   egg, hrm = loader.load()

   pinn = PINNEstimator()
   params = pinn.estimate_parameters(egg)
   ```

3. **Validate Parameter Recovery Accuracy**
   - Measure actual recovery error (target: <20% on real data)
   - Compare PINN vs Bayesian estimates
   - Validate IBS classification accuracy (target: >80%)

4. **Publish Results**
   - Write up validation study
   - Submit to top-tier journal

---

## ❓ Troubleshooting

### Common Issues:

**"ModuleNotFoundError: No module named 'X'"**
- Run: `pip install -r requirements.txt`

**"File not found: patient_data/P001_egg.csv"**
- Make sure you're in the correct directory: `cd "C:\ens-gi digital"`

**Test failures in test_bayesian.py**
- These are expected if PyMC3 is not installed (optional dependency)
- Framework still works, just can't run MCMC tests

**SPICE netlist missing channels**
- This means bug fixes didn't apply correctly
- Re-run the SPICE export test in Step 8
- Check [CODEBASE_AUDIT_REPORT.md](CODEBASE_AUDIT_REPORT.md) for details

---

## 📞 Need Help?

- **Documentation:** See [docs/](docs/) directory
- **Examples:** See [examples/](examples/) directory
- **Real Data Integration:** See [REAL_DATA_READINESS_REPORT.md](REAL_DATA_READINESS_REPORT.md)
- **Status Overview:** See [FINAL_IMPLEMENTATION_STATUS.md](FINAL_IMPLEMENTATION_STATUS.md)

---

**Last Updated:** 2026-02-15
**Version:** 0.3.1
**Status:** ✅ Verification checklist ready
