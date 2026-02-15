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

This implementation represents a **3-year phased development** as outlined in the research paper "Building a Gut Digital Twin."

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

# Install dependencies
pip install -r requirements.txt

# Optional: Install for development
pip install -e .
```

### Basic Usage

```python
from ens_gi_core import ENSGIDigitalTwin

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
from ens_gi_pinn import PINNEstimator

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
from ens_gi_bayesian import BayesianEstimator

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
| **Phase 1** | Mathematical Engine | 95% | ✅ Nearly Complete |
| **Phase 2** | Hardware Realization | 75% | 🟡 In Progress |
| **Phase 3** | Clinical Digital Twin | 85% | ✅ Nearly Complete |

### Phase 1: Mathematical Engine (Year 1) — 95% Complete ✅
- ✅ Extended HH model with multiple ion channels
- ✅ ICC pacemaker (FHN framework)
- ✅ Smooth muscle with Hill function
- ✅ Network architecture with gap junctions
- ✅ Parameter sweep for bifurcation analysis
- ✅ Python simulator with RK4 integration
- ⏳ Validation against Thomas-Bornstein AH neuron data (minor)

### Phase 2: Hardware Realization (Year 2) — 75% Complete 🟡
- ✅ **Verilog-A Standard Cell Library** (8 modules: Na, K, Ca, KCa, A-type K, Leak, Gap Junction, ICC)
- ✅ **SPICE netlist generation** (6 ion channels: Na, K, Ca, Leak, KCa, A-type K) 🆕
- ✅ **Behavioral subcircuit models** for ngspice compatibility 🆕
- ✅ **Automated validation script** (`validate_spice.py`) 🆕
- ✅ Memristive ion channel concept
- ⏳ **SPICE validation in ngspice** (script ready, awaiting execution)
- ⏳ 2D tissue simulation (100×100 ICC grid)
- ⏳ Wave propagation validation (3-12 mm/s)

**Recent Fixes (2026-02-15):** Fixed critical SPICE bugs - added missing Ca²⁺, KCa, and A-type K channel subcircuits. SPICE export now includes all 6 ion channel types.

### Phase 3: Clinical Digital Twin (Year 3) — 85% Complete ✅
- ✅ IBS-D, IBS-C, IBS-M pathology profiles
- ✅ Biomarker extraction (ICC freq, motility, spike rate)
- ✅ Clinical report generation
- ✅ **PINN framework implemented** (798 lines, physics-informed neural networks)
- ✅ **Bayesian inference framework implemented** (760 lines, PyMC3 integration)
- ✅ **Drug library with 7 FDA-approved drugs** (PK/PD modeling, virtual trials)
- ✅ **Patient data loader** (CSV support for EGG/HRM signals)
- ✅ **Clinical workflow integration** (parameter estimation → biomarkers → treatment)
- ✅ **Comprehensive test suite** (77 tests, >80% coverage)
- ⏳ **PINN/Bayesian validation on real patient data** (currently tested on synthetic data)
- ⏳ **Real clinical dataset integration** (synthetic data documented, awaiting open-source datasets)

**Note:** Current patient data (P001-P003) is synthetically generated for testing. Real clinical data from open-source datasets will be integrated post-validation. See `patient_data/README.md` for details.

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test module
pytest tests/test_pinn.py -v
```

---

## 📁 Project Structure

```
ens-gi-digital-twin/
├── ens_gi_core.py              # Core simulation engine (1,197 lines)
├── ens_gi_pinn.py              # Physics-Informed Neural Network (PINN)
├── ens_gi_bayesian.py          # Bayesian MCMC inference
├── ens_gi_drug_library.py      # Virtual drug trial system (TODO)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── IMPLEMENTATION_TODO.md      # Detailed implementation checklist
│
├── examples/
│   ├── clinical_parameter_estimation_workflow.py
│   ├── 01_basic_simulation.ipynb (TODO)
│   ├── 02_ibs_profiles.ipynb (TODO)
│   ├── 03_parameter_sweep.ipynb (TODO)
│   └── ... (more tutorials)
│
├── tests/
│   ├── test_ion_channels.py (TODO)
│   ├── test_pinn.py (TODO)
│   ├── test_bayesian.py (TODO)
│   └── ... (more tests)
│
├── verilog_a_library/ (TODO)
│   ├── NaV1_5.va
│   ├── Kv_delayed_rectifier.va
│   └── ... (hardware modules)
│
└── docs/
    ├── Building a Gut Digital Twin.pdf
    ├── api_reference.md (TODO)
    └── mathematical_model.md (TODO)
```

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

### Immediate Priorities (P0 - Critical)
- [x] PINN framework implementation
- [x] Bayesian inference framework
- [ ] PINN validation (<10% error on synthetic data)
- [ ] Bayesian MCMC validation (95% CI coverage)

### Short-term (P1 - High)
- [ ] Complete Verilog-A standard cell library
- [ ] Fix SPICE netlist generation (runnable in ngspice)
- [ ] Implement 2D tissue simulation
- [ ] Wave propagation validation

### Medium-term (P2 - Medium)
- [ ] Structured drug trial system
- [ ] Comprehensive test suite (>80% coverage)
- [ ] Documentation and tutorials
- [ ] Performance optimization (Numba JIT)

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

**Status**: Phase 3 in active development (50% complete)

---

**Last Updated**: 2026-02-14
#   E N S - G I - D i g i t a l - T w i n  
 