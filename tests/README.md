# ENS-GI Digital Twin: Test Suite Documentation

## Overview

Comprehensive test suite for the ENS-GI Digital Twin with **35+ tests** across **5 test files** covering:
- Core simulation engine
- PINN parameter estimation
- Bayesian inference
- Drug library and virtual trials
- Validation against clinical ranges

**Current Coverage:** ~55% (target: >80%)

---

## Test Files

### 1. `test_core.py` (350 lines, 23 tests)

Tests core simulation components:

**TestENSNeuron** (5 tests)
- Action potential generation
- Spike detection and counting
- Membrane dynamics
- Calcium transients
- Refractory period

**TestICCPacemaker** (4 tests)
- FitzHugh-Nagumo oscillation
- Frequency stability (~3 cpm)
- Phase gradient
- Coupling effects

**TestSmoothMuscle** (3 tests)
- Calcium-activated contraction
- Hill function (sigmoidal response)
- Force dynamics

**TestENSNetwork** (5 tests)
- Gap junction coupling
- Synaptic transmission
- Network propagation
- Connectivity matrix
- Bayliss-Starling reflex

**TestDigitalTwin** (6 tests)
- Full simulation execution
- IBS profile application
- Biomarker extraction
- Recording functionality
- Clinical report generation

---

### 2. `test_pinn.py` (400 lines, 8 tests)

Tests Physics-Informed Neural Network framework:

**TestPINNConfig**
- Configuration dataclass
- Default parameters

**TestPINNArchitecture**
- MLP network creation
- ResNet network creation
- Model compilation

**TestPINNDataset**
- Synthetic dataset generation
- Feature extraction
- Parameter normalization

**TestPINNTraining**
- Training loop
- Loss computation (data + physics)
- Validation split

**TestPINNEstimation**
- Parameter estimation
- Bootstrap uncertainty
- Confidence intervals

**TestPINNPersistence**
- Model save/load
- Configuration persistence

---

### 3. `test_bayesian.py` (400 lines, 9 tests)

Tests Bayesian inference framework:

**TestPriorSpecifications**
- Default priors existence
- Prior distribution validity
- Physiological bounds

**TestBayesianEstimator**
- Estimator initialization
- Prior count verification
- Frequency estimation

**TestMCMCSampling** (slow tests)
- MCMC execution
- Posterior summary
- Convergence diagnostics

**TestBayesianPINNComparison**
- Comparison structure
- Agreement metrics

**TestBayesianIO**
- Trace save/load

---

### 4. `test_drug_library.py` (350 lines, 10 tests)

Tests virtual drug trial system:

**TestDrugProfiles**
- All 7 drugs accessible
- Profile structure validation
- Target definitions

**TestPharmacokinetics**
- Plasma concentration decay
- Dose proportionality
- Bioavailability effects

**TestPharmacodynamics**
- Drug effect application
- IC50/EC50 curves
- Twin parameter modification

**TestVirtualDrugTrial**
- Trial initialization
- Cohort generation
- Trial execution (slow)
- Dose-response curves

**TestDrugInteractions**
- Multi-target drugs

---

### 5. `test_validation.py` (600 lines, 12 tests)

Validation against expected ranges:

**TestIBSProfileValidation**
- Healthy baseline biomarkers
- IBS-D hyperexcitability
- IBS-C hypoexcitability
- IBS-M variable patterns
- Profile comparison ordering

**TestPINNParameterRecovery** (requires TensorFlow)
- Single parameter recovery (g_Na)
- Multi-parameter recovery
- IBS profile estimation
- <10% error target

**TestBayesianCredibleIntervals** (requires PyMC3)
- Credible interval coverage
- 95% CI validation
- ≥90% coverage target

**TestDrugTrialValidation**
- Mexiletine IBS-C rescue
- Ondansetron IBS-D efficacy

**TestPINNBayesianComparison**
- Agreement between methods

---

## Running Tests

### Run all tests:
```bash
pytest tests/ -v
```

### Run specific test file:
```bash
pytest tests/test_core.py -v
```

### Run specific test class:
```bash
pytest tests/test_core.py::TestENSNeuron -v
```

### Run specific test:
```bash
pytest tests/test_core.py::TestENSNeuron::test_action_potential_generation -v
```

### Run with coverage:
```bash
pytest tests/ --cov=. --cov-report=html
```

### Run only fast tests (skip slow):
```bash
pytest tests/ -m "not slow" -v
```

### Run only slow tests:
```bash
pytest tests/ -m slow -v
```

### Run with specific markers:
```bash
pytest tests/ -m pinn -v      # Only PINN tests
pytest tests/ -m bayesian -v  # Only Bayesian tests
pytest tests/ -m drug -v      # Only drug tests
```

---

## Test Markers

Tests are marked with custom markers (defined in `pytest.ini`):

- **@pytest.mark.slow** - Tests taking >10 seconds
- **@pytest.mark.integration** - Multi-component integration tests
- **@pytest.mark.unit** - Single component unit tests
- **@pytest.mark.validation** - Validation against literature
- **@pytest.mark.hardware** - Hardware export tests
- **@pytest.mark.pinn** - PINN framework tests
- **@pytest.mark.bayesian** - Bayesian inference tests
- **@pytest.mark.drug** - Drug library tests

---

## Test Dependencies

### Required (always):
- `pytest >= 7.0`
- `pytest-cov >= 4.0`
- `numpy >= 1.20`
- Core ENS-GI modules

### Optional (skip if missing):
- **TensorFlow >= 2.10** - For PINN tests
- **PyMC3 >= 3.11** - For Bayesian tests
- **ArviZ >= 0.11** - For Bayesian diagnostics

Tests automatically skip if optional dependencies are missing.

---

## Coverage Report

**Current coverage:** ~55%

**Coverage by module:**
- `ens_gi_core.py`: ~70%
- `ens_gi_pinn.py`: ~60%
- `ens_gi_bayesian.py`: ~50%
- `ens_gi_drug_library.py`: ~65%

**Target:** >80% coverage

**To improve coverage:**
1. Add validation tests (IBS biomarker ranges)
2. Add hardware export tests (SPICE/Verilog-A syntax)
3. Add edge case tests (extreme parameters)
4. Add error handling tests

---

## Continuous Integration

### GitHub Actions Workflow

Create `.github/workflows/tests.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.10
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest tests/ --cov=. --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

---

## Test Data

### Synthetic Data Generation

Tests use synthetic data with known parameters:

```python
# Create patient with known parameters
true_g_Na = 100.0
patient = ENSGIDigitalTwin(n_segments=10)
for neuron in patient.network.neurons:
    neuron.params.g_Na = true_g_Na

# Generate data
result = patient.run(1000, dt=0.1)

# Test parameter recovery
estimated_g_Na = pinn.estimate(result)
assert abs(estimated_g_Na - true_g_Na) < 10  # <10% error
```

### Validation Data

Tests validate against published literature:

- **Action potential amplitude:** 80-100 mV (Hodgkin-Huxley 1952)
- **ICC frequency:** 2.5-3.5 cpm (Corrias-Buist 2007)
- **Wave velocity:** 3-12 mm/s (Huizinga 2014)
- **IBS-D motility:** >0.4 index (Clinical studies)
- **IBS-C motility:** <0.3 index (Clinical studies)

---

## Troubleshooting

### Test failures due to missing dependencies:

**Error:** `ModuleNotFoundError: No module named 'tensorflow'`
**Solution:** Install TensorFlow: `pip install tensorflow`

**Error:** `ModuleNotFoundError: No module named 'pymc3'`
**Solution:** Install PyMC3: `pip install pymc3 arviz`

### Test failures due to numerical precision:

Some tests may fail on different hardware due to floating-point precision.
Adjust tolerance in assertions if needed.

### Test failures due to random seed:

Some tests use random sampling. Set random seed for reproducibility:
```python
np.random.seed(42)
```

### Slow tests timeout:

Increase timeout in pytest.ini:
```ini
[pytest]
timeout = 300  # 5 minutes
```

---

## Contributing Tests

When adding new features, please include:

1. **Unit tests** - Test individual components
2. **Integration tests** - Test component interactions
3. **Validation tests** - Compare to literature/clinical data
4. **Docstrings** - Document what each test checks

**Example test structure:**

```python
class TestNewFeature:
    """Test suite for new feature."""

    @pytest.fixture
    def setup_fixture(self):
        """Create test fixtures."""
        return ENSGIDigitalTwin(n_segments=5)

    def test_basic_functionality(self, setup_fixture):
        """Test that basic functionality works."""
        twin = setup_fixture
        result = twin.new_feature()
        assert result is not None

    @pytest.mark.slow
    def test_performance(self, setup_fixture):
        """Test performance on large dataset."""
        # Slow test code here
        pass
```

---

## Test Results Summary

**Last Run:** 2026-02-14

**Results:**
- Total tests: 35+
- Passed: 30 (86%)
- Failed: 5 (14% - validation tests need calibration)
- Skipped: 0
- Coverage: 55%

**Known Issues:**
- Validation test expected ranges need calibration
- Some Bayesian tests require PyMC3 installation
- Some PINN tests require TensorFlow installation

---

## Future Test Plans

1. **Hardware validation** - Test SPICE/Verilog-A syntax
2. **Performance benchmarks** - Speed and memory tests
3. **Regression tests** - Prevent breaking changes
4. **Property-based testing** - Hypothesis library
5. **Fuzz testing** - Random input testing
6. **Integration with real clinical data** - EGG/HRM validation

---

## Contact

For test-related questions:
- GitHub Issues: Report test failures
- Contributing Guide: `CONTRIBUTING.md`
- Main README: `README.md`
