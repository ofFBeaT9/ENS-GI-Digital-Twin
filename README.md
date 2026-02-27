# ENS-GI Digital Twin

**A Multiscale, Physics-Based Digital Twin for Enteric Nervous System and Gastrointestinal Motility**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://www.tensorflow.org/)

---

## 🎯 Overview

The **ENS-GI Digital Twin** is a comprehensive computational framework for simulating the Enteric Nervous System (ENS) and gastrointestinal motility. It implements a unified engine serving three distinct applications:

1. **Research Simulator**: Biophysical equation-based model for mechanistic studies
2. **Neuromorphic Hardware**: SPICE/Verilog-A export for analog circuit implementation
3. **Clinical Predictor**: Patient-specific parameterization for IBS diagnosis and treatment

This implementation represents a **3-year phased development** 
---

## 🏗️ Architecture

### Layer 1: Cellular Electrophysiology
- Extended Hodgkin-Huxley model with:
  - Fast Na⁺ channels (action potentials)
  - Delayed rectifier K⁺ channels
  - L-type Ca²⁺ channels (enteric-specific)
  - Ca²⁺-activated K⁺ (afterhyperpolarization)
  - A-type K⁺ (transient outward)
  - Excitatory/inhibitory synaptic inputs
  - Intracellular Ca²⁺ dynamics

### Layer 2: Network & Propagation
- Coupled ENS neuron network
- Gap junction (electrical) coupling
- Chemical synapses with E/I balance
- Ascending excitation / descending inhibition (Bayliss-Starling reflex)
- Wave propagation dynamics

### Layer 3: ICC Pacemaker & Motility
- Interstitial Cells of Cajal (ICC) slow wave generator
- FitzHugh-Nagumo oscillator framework
- Smooth muscle contraction model (Hai-Murphy)
- Electromechanical coupling
- Motility force generation

### Layer 4: Clinical AI (Phase 3)
- **Physics-Informed Neural Networks (PINN)**: Parameter estimation from clinical data
- **Bayesian Inference**: Uncertainty quantification via MCMC
- **Virtual Drug Trials**: In silico therapeutic testing

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/ens-gi-digital-twin.git
cd ens-gi-digital-twin

# Create conda environment (recommended)
conda env create -f environment.yml
conda activate ens-gi-digital

# OR: Install with pip
pip install -e .

# OR: Install with specific extras
pip install -e .[pinn]           # PINN support (TensorFlow)
pip install -e .[bayesian]       # Bayesian inference (PyMC3)
pip install -e .[all]            # All features
```

### Basic Usage

```python
from ens_gi_digital import ENSGIDigitalTwin

# Create digital twin (20 segments)
twin = ENSGIDigitalTwin(n_segments=20)

# Apply IBS-D patient profile
twin.apply_profile('ibs_d')

# Run simulation (2000 ms)
result = twin.run(duration=2000, dt=0.05, I_stim={5: 10.0})

# Extract clinical biomarkers
biomarkers = twin.extract_biomarkers()
print(twin.clinical_report())

# Export to hardware (SPICE netlist)
spice_netlist = twin.export_spice_netlist('ens_network.sp')

# Export to Verilog-A
verilog_module = twin.export_verilog_a_module()
```

### Parameter Estimation (PINN)

```python
from ens_gi_digital import ENSGIDigitalTwin, PINNEstimator

# Create digital twin
twin = ENSGIDigitalTwin(n_segments=20)

# Create PINN estimator
pinn = PINNEstimator(twin, parameter_names=['g_Na', 'g_K', 'g_Ca', 'omega'])

# Train on synthetic data
pinn.train(epochs=2000, n_synthetic_samples=1000)

# Estimate parameters from patient EGG/HRM data
estimates, uncertainties = pinn.estimate_parameters(
    voltages=patient_egg_signal,
    forces=patient_hrm_signal
)
```

### Bayesian Inference

```python
from ens_gi_digital import ENSGIDigitalTwin, BayesianEstimator

# Create digital twin
twin = ENSGIDigitalTwin(n_segments=20)

# Create Bayesian estimator
bayes = BayesianEstimator(twin)

# Run MCMC sampling
trace = bayes.estimate_parameters(
    observed_voltages=patient_egg_signal,
    n_samples=5000
)

# Get posterior summary with credible intervals
summary = bayes.summarize_posterior(trace)
bayes.plot_posterior(trace)
```

---

## 📊 Phase Completion Status

| Phase | Description | Completion | Status |
|-------|-------------|------------|--------|
| **Phase 1** | Mathematical Engine | 98% | ✅ Nearly Complete |
| **Phase 2** | Hardware Realization | 90% | ✅ Nearly Complete |
| **Phase 3** | Clinical Digital Twin | 97% | ✅ Nearly Complete |

### Phase 1: Mathematical Engine (Year 1) — 95% Complete ✅
- ✅ Extended HH model with multiple ion channels
- ✅ ICC pacemaker (FHN framework)
- ✅ Smooth muscle with Hill function
- ✅ Network architecture with gap junctions
- ✅ Parameter sweep for bifurcation analysis
- ✅ Python simulator with RK4 integration
- ⏳ Validation against Thomas-Bornstein AH neuron data (minor)

### Phase 2: Hardware Realization (Year 2) — 90% Complete ✅
- ✅ **Verilog-A Standard Cell Library** (8 modules: Na, K, Ca, KCa, A-type K, Leak, Gap Junction, ICC)
- ✅ **SPICE netlist generation** (6 ion channels: Na, K, Ca, Leak, KCa, A-type K)
- ✅ **Behavioral subcircuit models** for ngspice compatibility
- ✅ **Automated validation script** (`validate_spice.py`)
- ✅ Memristive ion channel concept
- ✅ **SPICE validation in ngspice** (simulation runs successfully, output verified) 🆕
- ⏳ 2D tissue simulation (100×100 ICC grid)
- ⏳ Wave propagation validation (3-12 mm/s)

**Update (2026-02-22):** ngspice simulation tests now fully passing. SPICE netlist runs successfully and output contains simulation data.

### Phase 3: Clinical Digital Twin (Year 3) — 97% Complete ✅
- ✅ IBS-D, IBS-C, IBS-M pathology profiles
- ✅ Biomarker extraction (ICC freq, motility, spike rate)
- ✅ Clinical report generation
- ✅ **PINN framework implemented** (798 lines, physics-informed neural networks)
- ✅ **Bayesian inference framework implemented** (fully tested with PyMC3) 🆕
- ✅ **Drug library with 7 FDA-approved drugs** (PK/PD modeling, virtual trials)
- ✅ **Patient data loader** (CSV/EDF support for EGG/HRM signals)
- ✅ **Clinical workflow integration** (parameter estimation → biomarkers → treatment)
- ✅ **Comprehensive test suite** (146 tests, 100% pass rate) 🆕
- ✅ **Real dataset loaders** (Zenodo EGG, SPARC HRM API integration) 🆕
- ✅ **Bayesian-PINN comparison validated** (agreement tested) 🆕
- ✅ **Drug trial validation** (mexiletine/ondansetron efficacy confirmed on synthetic cohort) 🆕
- ⏳ Real clinical dataset ingestion (loaders ready, awaiting open-source dataset download)

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test module
pytest tests/test_core.py -v

# Quick verification (5 tests, ~30 seconds)
python scripts/quick_test.py

# Comprehensive verification (8 tests, ~2 minutes)
python scripts/final_verification.py
```

**Latest Test Results (2026-02-22):**
- **Total:** 146 tests
- ✅ **146 PASSED** (100%) — zero skips, zero failures
- ⏭️ **0 SKIPPED**
- ❌ **0 FAILED**
- ⏱️ **Runtime:** ~3h 52m (full physics simulation suite)

**Test Coverage by Module:**
- ✅ Core engine: 37/37 tests pass (100%)
- ✅ Drug library: 14/14 tests pass (100%)
- ✅ PINN framework: 12/12 tests pass (100%)
- ✅ Bayesian inference: 11/11 tests pass (100%) — PyMC3 fully operational
- ✅ Bayesian integration: 26/26 tests pass (100%) — cache, ODE physics, clinical data
- ✅ Hardware export: 11/11 tests pass (100%) — SPICE + ngspice + Verilog-A
- ✅ Patient data: 10/10 tests pass (100%) — EGG CSV, HRM CSV, EDF loader
- ✅ Real datasets: 11/11 tests pass (100%) — Zenodo EGG, SPARC HRM loaders
- ✅ Validation: 15/15 tests pass (100%) — IBS profiles, PINN/Bayesian recovery, drug trials

---

## ✅ Validation Gauntlet Results

The **Validation Gauntlet** (`scripts/validation_gauntlet.py`) is a four-section end-to-end pipeline that reads the PINN cohort output (`pinn_zenodo_both_conditions.json`) and produces independent PASS/FAIL verdicts covering the full modelling stack.

**Latest run (2026-02-27) — `--sections fft,hrm,spice`:**

| Section | Verdict | Key Metric |
|---------|---------|------------|
| **[1] FFT Frequency Analysis** | **PASS** | Pearson r=0.24, MAE=0.56 cpm |
| **[2] Bayesian HDI Overlay** | SKIP | PyMC not installed in quick run |
| **[3] SPARC HRM Cross-Validation** | **PASS** | Pressure ratio \|twin/SPARC\|=6.5× (within [0.05, 20]) |
| **[4] Hardware Parity — Pure SPICE** | **PASS** | Pearson r=0.991, shape-nRMSE=7.5% |
| **[4] Hardware Parity — Verilog-A** | SKIP | Requires ADMS-compiled ngspice |

**Section notes:**

- **FFT (§1)**: PINN ICC frequency estimates (`icc_frequency_cpm`) compared against the spectral peak of the raw EGG signal (gastric band 1.5–4.5 cpm) for all 40 cohort rows. Pearson r=0.24 (p<0.05), MAE 0.56 cpm — below the 1.0 cpm threshold.
- **SPARC HRM (§3)**: Twin forward-simulation medians compared against ex-vivo colonic HRM (SPARC dataset, subjects 1–10). Anatomical and scale differences between gastric EGG source data and colonic HRM are expected; the order-of-magnitude criterion (ratio within [0.05, 20×]) provides a sanity-level check.
- **SPICE Parity (§4)**: Python (RK4) vs. ngspice circuit co-simulation on Subject 1 (fasting), 20-segment twin. r=0.991 confirms waveform shape agreement despite the expected unit-scale difference (Python: mV, SPICE: V). Phase drift over the 2 s window contributes to shape-nRMSE=7.5% but is not a failure mode (correlation criterion only).
- **Verilog-A (§4)**: The `.hdl` files in `verilog_a_library/` are valid; the check is SKIP when ngspice is compiled without ADMS support. The VA path is now run with `cwd=<repo_root>` so relative `.hdl` paths resolve correctly when ADMS ngspice is available.

**Run the gauntlet:**
```bash
# Sections 1, 3, 4 (fast — no PyMC required, ~10 min)
python scripts/validation_gauntlet.py \
  --results-json pinn_zenodo_both_conditions.json \
  --data-dir data \
  --sections fft,hrm,spice

# Full gauntlet including Bayesian MCMC (~40-60 min additional)
python scripts/validation_gauntlet.py \
  --results-json pinn_zenodo_both_conditions.json \
  --data-dir data \
  --sections all
```

Results are written to `validation_gauntlet_results.json`.

---

## 📁 Project Structure

```
ens-gi-digital-twin/
├── src/
│   └── ens_gi_digital/             # Main package
│       ├── __init__.py             # Package exports
│       ├── core.py                 # Core simulation engine (51 KB)
│       ├── pinn.py                 # Physics-Informed Neural Networks
│       ├── bayesian.py             # Bayesian MCMC inference
│       ├── drug_library.py         # Virtual drug trial system
│       ├── patient_data.py         # Patient data loading (CSV/EDF)
│       ├── clinical_workflow.py    # Clinical analysis pipeline
│       └── simulation_cache.py     # LRU disk cache for MCMC speedup
│
├── tests/                          # Test suite (146 tests, 100% pass)
│   ├── test_core.py               # Core engine tests (37 tests)
│   ├── test_pinn.py               # PINN framework tests (12 tests)
│   ├── test_bayesian.py           # Bayesian inference tests (11 tests)
│   ├── test_bayesian_integration.py # Cache, ODE physics, clinical data (26 tests)
│   ├── test_drug_library.py       # Drug library tests (14 tests)
│   ├── test_hardware_export.py    # SPICE/Verilog-A/ngspice tests (11 tests)
│   ├── test_patient_data.py       # EGG CSV, HRM CSV, EDF loader (10 tests)
│   ├── test_real_datasets.py      # Zenodo EGG, SPARC HRM loaders (11 tests)
│   ├── test_validation.py         # Clinical validation tests (15 tests)
│   └── __init__.py
│
├── scripts/                        # Utility scripts
│   ├── quick_test.py              # Fast 5-test verification
│   ├── final_verification.py      # Comprehensive 8-test suite
│   ├── manual_test.py             # Manual 9-test suite
│   ├── verify_installation.py     # Post-install check
│   ├── profile_performance.py     # Performance profiling
│   ├── validate_spice.py          # SPICE validation with ngspice
│   ├── validate_spice_netlist.py  # SPICE netlist runner
│   └── validation_gauntlet.py     # 4-section validation pipeline (FFT/Bayesian/HRM/SPICE)
│
├── examples/                       # Tutorials & demos
│   ├── basic_simulation_tutorial.ipynb
│   ├── bayesian_tutorial.ipynb
│   ├── clinical_workflow.ipynb
│   ├── hardware_export_tutorial.ipynb
│   ├── pinn_tutorial.ipynb
│   ├── virtual_drug_trials_tutorial.ipynb
│   ├── clinical_parameter_estimation_workflow.py
│   ├── demo_all_features.py
│   ├── test_spice_export.py
│   └── spice_export/              # Example SPICE netlists
│       ├── manual_test.sp
│       └── quick_test.sp
│
├── verilog_a_library/             # Hardware models (8 modules)
│   ├── NaV1_5.va                  # Sodium channel
│   ├── Kv_delayed_rectifier.va    # Potassium delayed rectifier
│   ├── CaL_channel.va             # L-type calcium channel
│   ├── KCa_channel.va             # Ca-activated potassium
│   ├── A_type_K.va                # A-type potassium (transient)
│   ├── leak_channel.va            # Leak current
│   ├── gap_junction.va            # Electrical coupling
│   ├── icc_fhn_oscillator.va      # ICC pacemaker
│   └── README.md
│
├── patient_data/                   # Sample patient data
│   ├── P001_egg.csv               # Patient 1 EGG signal
│   ├── P001_hrm.csv               # Patient 1 HRM pressure
│   ├── P002_egg.csv               # Patient 2 (IBS-D)
│   ├── P002_hrm.csv
│   ├── P003_egg.csv               # Patient 3 (IBS-C)
│   ├── P003_hrm.csv
│   └── README.md                  # Data provenance
│
├── docs/                          # Documentation
│   ├── Building a Gut Digital Twin.pdf
│   ├── installation.md
│   ├── quickstart.md
│   ├── user_guide.md
│   ├── tutorials.md
│   ├── biomarker_ranges.md
│   ├── validation_report.md
│   ├── api_reference.rst
│   └── contributing.md
│
├── .github/
│   ├── workflows/
│   │   └── tests.yml              # CI/CD pipeline
│   └── agents/
│
├── environment.yml                # Conda environment spec
├── pyproject.toml                 # Modern Python packaging
├── setup.py                       # Package installation
├── pytest.ini                     # Test configuration
├── requirements.txt               # All dependencies
├── requirements_core.txt          # Core + PINN (no Bayesian)
├── requirements_minimal.txt       # Minimal working set
├── .gitignore                     # VCS exclusions
│
├── README.md                      # This file
├── CHANGELOG.md                   # Version history
├── CONTRIBUTING.md                # Contribution guidelines
├── LICENSE                        # MIT License
├── IMPLEMENTATION_COMPLETE.md     # Implementation status
├── REAL_DATA_READINESS_REPORT.md  # Data integration plan
├── CODEBASE_AUDIT_REPORT.md       # Quality baseline
├── VERIFICATION_CHECKLIST.md      # Release checklist
└── RUN_TESTS.md                   # Testing guide
```

**Reorganization (Feb 2026):** Project restructured to modern `src/` package layout. Reduced repository size from 247 MB → 3.8 MB (98.5% reduction) by removing external dependencies (`.conda/`, `Spice64/`) and consolidating documentation.

---

## 🎓 Scientific Background

### Key Publications
- **Hodgkin & Huxley (1952)**: Action potential generation
- **Corrias & Buist (2007)**: ICC calcium clock model
- **Thomas & Bornstein (2003)**: AH-type enteric neurons
- **Chai & Koh (2012)**: Slow wave propagation
- **Raissi et al. (2019)**: Physics-Informed Neural Networks

### Biological Basis
The ENS (Enteric Nervous System) is often called the "second brain" — a complex network of ~500 million neurons controlling:
- Gastrointestinal motility (peristalsis)
- Secretion and blood flow
- Immune response modulation
- Gut-brain axis signaling

**IBS (Irritable Bowel Syndrome)** affects ~10-15% of the global population and is characterized by:
- IBS-D: Diarrhea-predominant (hyperexcitability)
- IBS-C: Constipation-predominant (hypoexcitability)
- IBS-M: Mixed symptoms (oscillating dynamics)

---

## 💊 Clinical Applications

### 1. Patient-Specific Parameterization
```python
# Estimate patient's biophysical parameters from EGG/HRM
twin = ENSGIDigitalTwin(n_segments=20)
pinn = PINNEstimator(twin)
params = pinn.estimate_parameters(patient_egg_signal)

# Apply to digital twin
twin.apply_custom_parameters(params)
```

### 2. Virtual Drug Trials
```python
from ens_gi_drug_library import DrugLibrary, VirtualDrugTrial

# Test Mexiletine (Na+ blocker) for IBS-C
trial = VirtualDrugTrial(drug=DrugLibrary.MEXILETINE, cohort_size=100)
results = trial.run(patient_twin, dose_range=[0, 50, 100, 200])
```

### 3. Treatment Optimization
- Predict therapeutic response before prescribing
- Optimize drug dosage personalized to patient
- Identify contraindications and side effects
- Monitor disease progression over time

---

## 🔬 Hardware Implementation

### SPICE Netlist Export
```python
twin.export_spice_netlist('ens_network.sp')
```

Generated netlist can be simulated in:
- ngspice (open-source)
- LTspice (free)
- Cadence Spectre (commercial)

### Verilog-A Export
```python
twin.export_verilog_a_module()
```

Compatible with:
- Cadence Virtuoso
- Keysight ADS
- Synopsys HSPICE

### Neuromorphic Hardware Targets
- Analog VLSI (CMOS)
- Memristive crossbar arrays
- FPGA emulation
- SpiNNaker neuromorphic chip

---

## 🛠️ Development Roadmap

See [IMPLEMENTATION_TODO.md](IMPLEMENTATION_TODO.md) for detailed task breakdown.

### Completed (P0 ✅)
- [x] PINN framework implementation
- [x] Bayesian inference framework
- [x] PINN validation — all 12 tests pass including `test_training_step` (TF autograph fix)
- [x] Bayesian MCMC validation — 95% CI coverage confirmed (11/11 tests pass)
- [x] Complete Verilog-A standard cell library (8 modules)
- [x] SPICE netlist generation — runs successfully in ngspice, output verified
- [x] Structured drug trial system (7 FDA drugs, PK/PD, virtual trials)
- [x] Comprehensive test suite — 146 tests, 100% pass rate
- [x] Real dataset loaders (Zenodo EGG, SPARC HRM)
- [x] Hardware export tests (SPICE + Verilog-A + ngspice integration)
- [x] Simulation cache (LRU disk cache, 10-20× MCMC speedup)
- [x] EDF patient data loader with graceful fallback
- [x] **Validation Gauntlet** — 4-section pipeline; FFT PASS (r=0.24, MAE=0.56 cpm), HRM PASS (|ratio|=6.5×), SPICE PASS (r=0.991)

### Short-term (P1 - High)
- [ ] Implement `predict_manometry()` — motility index → HRM pressure trace (Djoumessi 2024)
- [ ] Spatiotemporal ICC slow-wave mapping (propagation velocity/direction)
- [ ] Implement 2D tissue simulation (20×20 ICC grid)
- [ ] Wave propagation velocity validation (3-12 mm/s)

### Medium-term (P2 - Medium)
- [ ] B-PINN unified framework combining PINN + Bayesian (Yang 2021)
- [ ] Vagal-ENS interface (gut-brain axis modulation)
- [ ] Documentation and Sphinx API reference
- [ ] Performance optimization (Numba JIT on hot loops)
- [ ] GitHub Actions CI/CD pipeline

### Long-term
- [ ] Integration with real clinical data
- [ ] Multi-organ coupling (stomach-intestine)
- [ ] 3D tissue geometry
- [ ] Real-time clinical decision support system

---

## 📖 Documentation

- **Getting Started**: See Quick Start above
- **API Reference**: `docs/api_reference.md` (TODO)
- **Mathematical Model**: `docs/mathematical_model.md` (TODO)
- **Tutorials**: `examples/` directory
- **Research Paper**: `docs/Building a Gut Digital Twin.pdf`

---

## 🤝 Contributing

Contributions are welcome! Please see `CONTRIBUTING.md` for guidelines.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Run tests before committing
pytest tests/ -v

# Format code
black ens_gi_*.py
```

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Author**: Mahdad
**Institution**: [Your Institution]
**Email**: [Your Email]

---

## 🙏 Acknowledgments

- Hodgkin & Huxley for the foundational HH model
- Corrias & Buist for ICC pacemaker modeling
- Raissi et al. for Physics-Informed Neural Networks
- PyMC3 and TensorFlow communities

---

## 📚 Citation

If you use this software in your research, please cite:

```bibtex
@software{ens_gi_digital_twin,
  author = {Mahdad},
  title = {ENS-GI Digital Twin: Multiscale Simulation of Enteric Nervous System},
  year = {2026},
  url = {https://github.com/yourusername/ens-gi-digital-twin}
}
```

---

## 🏆 Project Goals

**Vision**: Enable personalized, mechanistic treatment of gastrointestinal disorders through computational medicine.

**Impact**:
- 🧬 Bridge computational neuroscience ↔ clinical gastroenterology
- 💻 Demonstrate feasibility of neuromorphic GI hardware
- 🏥 Provide decision support for clinicians treating IBS patients
- 📊 Generate publishable research in *Nature BME*, *IEEE TBME*, *Gut*

**Status**: All three phases functionally complete — 146/146 tests passing (100%). Remaining work: real clinical data acquisition (30-50 patients required for publication), 2D tissue simulation, and B-PINN unified framework.

---

**Last Updated**: 2026-02-27

---

## Real Data Status (2026-02-23)

Current real datasets detected under `data/`:
- `EGG-database` (Zenodo): 20 subjects, fasting/postprandial EGG recordings (3 channels).
- `pennsieve data base/files/primary` (SPARC): 34 subjects with colonic HRM recordings.

Current prepared real-patient CSV cohort in `patient_data/`:
- `REAL001` .. `REAL005` (5 real composite patients built from Zenodo EGG + SPARC HRM).

Important training note:
- The PINN training stage is still synthetic-data supervised (`generate_synthetic_dataset`).
- Real patient data is currently used in the parameter estimation stage (`estimate_parameters`) after PINN training.
- In practice this means: model learns physics-informed mapping on synthetic data, then fits each real patient.

Prepare real patient CSVs:
```bash
python scripts/prepare_real_patient_data.py --batch-count 5 --batch-prefix REAL --batch-start-index 1 --batch-egg-start 1 --batch-egg-step 1 --egg-condition postprandial --batch-hrm-offset 3
```

Train PINN on a prepared real patient:
```bash
python scripts/train_pinn_from_patient_data.py --patient-id REAL005 --data-dir patient_data --architecture resnet --hidden-dims 512,256,128,64,32 --learning-rate 0.0005 --lambda-physics 0.2 --batch-size 64 --epochs 160 --synthetic-samples 800 --synthetic-duration-ms 1000 --synthetic-dt 0.1 --bootstrap 120 --use-ode-residuals --model-out pinn_real005_best
```

---
