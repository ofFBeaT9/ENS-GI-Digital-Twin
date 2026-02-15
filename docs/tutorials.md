# Tutorials

Interactive Jupyter notebook tutorials for hands-on learning.

---

## Available Tutorials

All tutorials are located in the `examples/` directory as Jupyter notebooks (`.ipynb` files).

### 1. Basic Simulation Tutorial

**File:** `examples/basic_simulation_tutorial.ipynb`

**Topics:**
- Creating digital twins
- Running simulations
- Visualizing results (voltage, calcium, ICC, force)
- Extracting biomarkers
- Parameter sweeps (sensitivity analysis)
- IBS profile comparison
- Spike analysis

**Level:** Beginner

**Duration:** 30 minutes

---

### 2. Clinical Parameter Estimation Workflow

**File:** `examples/clinical_workflow.ipynb`

**Topics:**
- Synthetic patient data generation
- PINN parameter estimation (fast)
- Bayesian refinement with uncertainty
- Virtual drug trials
- Clinical report generation
- Complete end-to-end pipeline

**Level:** Intermediate

**Duration:** 45 minutes

---

### 3. PINN Tutorial

**File:** `examples/pinn_tutorial.ipynb`

**Topics:**
- What is a PINN?
- Training a PINN
- Parameter estimation (solving inverse problem)
- Uncertainty quantification (bootstrap)
- Model validation
- Comparison with Bayesian methods

**Level:** Advanced

**Duration:** 60 minutes

---

### 4. Bayesian Inference Tutorial

**File:** `examples/bayesian_tutorial.ipynb`

**Topics:**
- Bayes' theorem fundamentals
- Prior distributions
- MCMC sampling (NUTS algorithm)
- Convergence diagnostics (R-hat, ESS)
- Credible intervals (95% CI)
- Posterior analysis

**Level:** Advanced

**Duration:** 60 minutes

---

### 5. Virtual Drug Trials Tutorial

**File:** `examples/virtual_drug_trials_tutorial.ipynb`

**Topics:**
- Drug library (7 FDA-approved drugs)
- Pharmacokinetics (PK curves)
- Single-patient trials
- Multi-drug comparison
- Dose-response curves
- Cohort-based clinical trials
- Multi-target drug effects

**Level:** Intermediate

**Duration:** 45 minutes

---

### 6. Hardware Export Tutorial

**File:** `examples/hardware_export_tutorial.ipynb`

**Topics:**
- Verilog-A fundamentals
- SPICE netlist generation
- Ion channel library (8 modules)
- Network topology export
- ngspice/Spectre workflow
- Hardware resource estimation
- FPGA vs ASIC trade-offs

**Level:** Advanced

**Duration:** 60 minutes

---

## Running Tutorials

### Prerequisites

```bash
pip install jupyter matplotlib seaborn
```

### Launch Jupyter

```bash
cd examples/
jupyter notebook
```

### In Browser

Navigate to desired tutorial and run cells sequentially.

---

## Tutorial Learning Path

**For Beginners:**
1. Basic Simulation Tutorial
2. Clinical Workflow Tutorial
3. Virtual Drug Trials Tutorial

**For Researchers:**
1. Basic Simulation Tutorial
2. PINN Tutorial
3. Bayesian Tutorial

**For Hardware Engineers:**
1. Basic Simulation Tutorial
2. Hardware Export Tutorial

**For Clinicians:**
1. Basic Simulation Tutorial
2. Clinical Workflow Tutorial
3. Virtual Drug Trials Tutorial

---

## Additional Resources

### Python Demos

Located in `examples/` directory:

- `clinical_parameter_estimation_workflow.py` - Complete clinical pipeline (Python)
- `test_spice_export.py` - Hardware export demonstration
- `demo_all_features.py` - Comprehensive feature showcase

Run with:
```bash
python examples/clinical_parameter_estimation_workflow.py
```

### API Examples

See [API Reference](api_reference.rst) for code snippets.

---

## Troubleshooting

**Jupyter not installed?**
```bash
pip install jupyter
```

**Kernel crashes?**
- Reduce network size (`n_segments=5`)
- Increase available RAM

**Missing dependencies?**
```bash
pip install -r requirements.txt
```

**TensorFlow not found?**
```bash
pip install tensorflow  # For PINN tutorial
```

**PyMC3 not found?**
```bash
pip install pymc3 arviz  # For Bayesian tutorial
```

---

## Contributing Tutorials

Have a tutorial idea? See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

Tutorial checklist:
- [ ] Clear learning objectives
- [ ] Step-by-step instructions
- [ ] Working code examples
- [ ] Visualizations
- [ ] Summary of key points
- [ ] Next steps / further reading
