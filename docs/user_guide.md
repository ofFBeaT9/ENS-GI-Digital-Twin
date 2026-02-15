# User Guide

Comprehensive guide to using ENS-GI Digital Twin.

## Table of Contents

1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [Creating Simulations](#creating-simulations)
4. [IBS Modeling](#ibs-modeling)
5. [Parameter Estimation](#parameter-estimation)
6. [Virtual Drug Trials](#virtual-drug-trials)
7. [Hardware Export](#hardware-export)
8. [Advanced Topics](#advanced-topics)

---

## Introduction

The ENS-GI Digital Twin provides three integrated applications from a single codebase:

1. **Research Simulator** - Explore parameter spaces and network dynamics
2. **Neuromorphic Hardware** - Export to SPICE/Verilog-A for silicon implementation
3. **Clinical Predictor** - Patient-specific parameter estimation with AI

---

## Core Concepts

### Biological Background

**Enteric Nervous System (ENS):**
- "Second brain" - ~100 million neurons
- Controls GI motility, secretion, blood flow
- Semi-autonomous from CNS

**Key Components:**
- **Neurons:** Hodgkin-Huxley model with 6 ion channels
- **ICC (Interstitial Cells of Cajal):** Pacemaker cells (~3 cycles/min)
- **Smooth Muscle:** Calcium-activated contraction

### Mathematical Model

**Neurons:** Extended Hodgkin-Huxley

```
C_m * dV/dt = -I_Na - I_K - I_Ca - I_KCa - I_A - I_L + I_ext + I_syn + I_gap
```

**ICC:** FitzHugh-Nagumo oscillator

```
dV_icc/dt = V_icc - V_icc³/3 - W_icc + I
dW_icc/dt = ω * (V_icc + a - b*W_icc)
```

**Smooth Muscle:** Hai-Murphy model

```
F = [Ca²⁺]^n / (K_half^n + [Ca²⁺]^n)
```

---

## Creating Simulations

### Basic Workflow

1. Create digital twin
2. Configure parameters
3. Run simulation
4. Extract biomarkers
5. Analyze results

### Example

```python
from ens_gi_core import ENSGIDigitalTwin

# Create
twin = ENSGIDigitalTwin(n_segments=10)

# Configure
twin.apply_profile('healthy')

# Run
result = twin.run(duration=2000, dt=0.05)

# Extract
biomarkers = twin.extract_biomarkers()

# Analyze
print(twin.clinical_report())
```

See [Quick Start Guide](quickstart.md) for more examples.

---

## IBS Modeling

### IBS Subtypes

**IBS-D (Diarrhea-predominant):**
- Increased Na+ conductance → Hyperexcitability
- Elevated motility index
- Faster transit

**IBS-C (Constipation-predominant):**
- Reduced Na+ conductance → Hypoexcitability
- Decreased motility index
- Slower transit

**IBS-M (Mixed):**
- Variable dynamics
- Alternating patterns

### Usage

```python
twin = ENSGIDigitalTwin(n_segments=10)
twin.apply_profile('ibs_d')  # or 'ibs_c', 'ibs_m'
```

---

## Parameter Estimation

### PINN (Physics-Informed Neural Networks)

Fast parameter estimation combining data with physics:

```python
from ens_gi_pinn import PINNEstimator, PINNConfig

pinn = PINNEstimator(twin, PINNConfig())
estimates = pinn.estimate_parameters(voltages, forces, calcium)
```

See [PINN Tutorial](../examples/pinn_tutorial.ipynb) for details.

### Bayesian Inference

Rigorous uncertainty quantification:

```python
from ens_gi_bayesian import BayesianEstimator

bayes = BayesianEstimator(twin)
trace = bayes.estimate_parameters(voltages)
summary = bayes.summarize_posterior(trace)
```

See [Bayesian Tutorial](../examples/bayesian_tutorial.ipynb) for details.

---

## Virtual Drug Trials

Test drugs in silico before clinical trials:

```python
from ens_gi_drug_library import DrugLibrary, apply_drug

# Apply drug
apply_drug(twin, DrugLibrary.LUBIPROSTONE, dose_mg=24)

# Run trial
from ens_gi_drug_library import VirtualDrugTrial
trial = VirtualDrugTrial(drug=DrugLibrary.LUBIPROSTONE, cohort_size=30)
results = trial.run_trial(doses_mg=[12, 24, 48])
```

See [Virtual Drug Trials Tutorial](../examples/virtual_drug_trials_tutorial.ipynb).

---

## Hardware Export

Export to SPICE/Verilog-A for neuromorphic chips:

```python
# SPICE netlist
twin.export_spice_netlist('network.sp', use_verilog_a=False)

# Verilog-A netlist
twin.export_spice_netlist('network_va.sp', use_verilog_a=True)

# Standalone module
va_module = twin.export_verilog_a_module()
```

See [Hardware Export Tutorial](../examples/hardware_export_tutorial.ipynb).

---

## Advanced Topics

### Custom Parameters

```python
# Modify individual neuron
twin.network.neurons[0].params.g_Na = 150.0

# Modify ICC frequency
twin.network.icc.omega = 0.012  # Faster pacing
```

### Custom Connectivity

```python
# Add custom gap junction
twin.network.neurons[0].add_gap_junction(
    target_idx=5,
    strength=0.5
)
```

### Performance Optimization

```python
# Larger timestep (faster, less accurate)
result = twin.run(1000, dt=0.5)

# Fewer segments
twin = ENSGIDigitalTwin(n_segments=5)
```

---

## References

For more information:
- [API Reference](api_reference.rst)
- [Tutorials](tutorials.md)
- [Validation Report](validation_report.md)
