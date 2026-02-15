# Quick Start Guide

Get started with ENS-GI Digital Twin in 5 minutes!

---

## 1. Basic Simulation

Create and run your first simulation:

```python
from ens_gi_core import ENSGIDigitalTwin

# Create digital twin with 10 gut segments
twin = ENSGIDigitalTwin(n_segments=10)

# Apply healthy profile
twin.apply_profile('healthy')

# Run simulation for 2000 ms
result = twin.run(duration=2000, dt=0.05, I_stim={5: 10.0})

print(f"Simulation complete!")
print(f"  Time points: {len(result['time'])}")
print(f"  Neurons: {result['voltages'].shape[1]}")
```

**Output:**
```
Simulation complete!
  Time points: 40000
  Neurons: 10
```

---

## 2. Extract Biomarkers

Get clinically relevant metrics:

```python
# Extract biomarkers
biomarkers = twin.extract_biomarkers()

print("Clinical Biomarkers:")
for key, value in biomarkers.items():
    print(f"  {key}: {value:.3f}")
```

**Output:**
```
Clinical Biomarkers:
  motility_index: 0.345
  icc_frequency_cpm: 3.12
  spike_rate_per_neuron: 0.823
  mean_voltage: -58.42
  mean_force: 0.234
```

---

## 3. Generate Clinical Report

```python
# Generate human-readable report
report = twin.clinical_report()
print(report)
```

**Output:**
```
ENS-GI DIGITAL TWIN - CLINICAL REPORT
=====================================
Profile: healthy
Segments: 10
Simulation Duration: 2000.00 ms

BIOMARKERS:
  ICC Frequency: 3.12 cpm (normal: 2.5-3.5)
  Motility Index: 0.345 (healthy range)
  Spike Rate: 0.823 Hz/neuron
  Mean Voltage: -58.42 mV
  Mean Force: 0.234

INTERPRETATION:
  ✓ Normal ICC pacemaker activity
  ✓ Healthy motility pattern
  ✓ Regular neural firing
```

---

## 4. Visualize Results

```python
import matplotlib.pyplot as plt

# Plot voltage traces
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

t = result['time'][:2000]  # First 100 ms

# Neural activity
axes[0].plot(t, result['voltages'][:2000, 0])
axes[0].set_ylabel('Voltage (mV)')
axes[0].set_title('Neural Activity')

# ICC slow wave
axes[1].plot(t, result['icc_currents'][:2000], color='purple')
axes[1].set_ylabel('ICC Current (pA)')
axes[1].set_title('ICC Pacemaker')

# Smooth muscle force
axes[2].plot(t, result['forces'][:2000, 0], color='green')
axes[2].set_ylabel('Force')
axes[2].set_xlabel('Time (ms)')
axes[2].set_title('Muscle Contraction')

plt.tight_layout()
plt.show()
```

---

## 5. IBS Simulation

Compare healthy vs IBS profiles:

```python
profiles = ['healthy', 'ibs_d', 'ibs_c']
results = {}

for profile in profiles:
    twin = ENSGIDigitalTwin(n_segments=10)
    twin.apply_profile(profile)
    twin.run(1500, dt=0.05, verbose=False)
    results[profile] = twin.extract_biomarkers()

# Compare motility
for profile in profiles:
    motility = results[profile]['motility_index']
    print(f"{profile.upper():<10} Motility: {motility:.3f}")
```

**Output:**
```
HEALTHY    Motility: 0.345
IBS_D      Motility: 0.487  (↑ hyperexcitable)
IBS_C      Motility: 0.198  (↓ hypoexcitable)
```

---

## 6. Parameter Sweep

Explore how parameters affect motility:

```python
import numpy as np

g_Na_values = np.linspace(80, 160, 9)
motility_values = []

for g_Na in g_Na_values:
    twin = ENSGIDigitalTwin(n_segments=8)
    for neuron in twin.network.neurons:
        neuron.params.g_Na = g_Na

    twin.run(1000, dt=0.1, verbose=False)
    bio = twin.extract_biomarkers()
    motility_values.append(bio['motility_index'])

# Plot
plt.plot(g_Na_values, motility_values, 'o-', linewidth=2)
plt.xlabel('g_Na (mS/cm²)')
plt.ylabel('Motility Index')
plt.title('Excitability vs Motility')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 7. Virtual Drug Trial

Test a drug in silico:

```python
from ens_gi_drug_library import DrugLibrary, apply_drug

# Create IBS-C patient
patient = ENSGIDigitalTwin(n_segments=10)
patient.apply_profile('ibs_c')

# Baseline
patient.run(1500, dt=0.05, verbose=False)
baseline = patient.extract_biomarkers()

# Apply Lubiprostone (IBS-C drug)
patient_drug = ENSGIDigitalTwin(n_segments=10)
patient_drug.apply_profile('ibs_c')
apply_drug(patient_drug, DrugLibrary.LUBIPROSTONE, dose_mg=24)

patient_drug.run(1500, dt=0.05, verbose=False)
post_drug = patient_drug.extract_biomarkers()

# Compare
improvement = (post_drug['motility_index'] - baseline['motility_index']) / baseline['motility_index'] * 100

print(f"Baseline motility: {baseline['motility_index']:.3f}")
print(f"Post-drug motility: {post_drug['motility_index']:.3f}")
print(f"Improvement: {improvement:.1f}%")
```

---

## 8. Hardware Export

Export to SPICE for hardware simulation:

```python
# Export SPICE netlist
twin = ENSGIDigitalTwin(n_segments=5)
twin.export_spice_netlist('my_network.sp', use_verilog_a=False)

print("SPICE netlist exported!")
print("Run with: ngspice my_network.sp")
```

---

## 9. AI Parameter Estimation

Estimate parameters from clinical data:

```python
from ens_gi_pinn import PINNEstimator, PINNConfig

# Create patient with unknown parameters
patient = ENSGIDigitalTwin(n_segments=10)
result = patient.run(1500, dt=0.1, verbose=False)

# Train PINN
pinn = PINNEstimator(patient, PINNConfig(), parameter_names=['g_Na', 'g_K'])
dataset = pinn.generate_synthetic_dataset(n_samples=200)
pinn.train(dataset['features'], dataset['parameters'], epochs=500, verbose=False)

# Estimate
estimates = pinn.estimate_parameters(
    result['voltages'],
    result['forces'],
    result['calcium']
)

print("Parameter Estimates:")
for param, est in estimates.items():
    print(f"  {param}: {est['mean']:.2f} ± {est['std']:.2f}")
```

---

## 10. Jupyter Notebooks

For interactive tutorials, see:

- `examples/basic_simulation_tutorial.ipynb` - Introduction
- `examples/clinical_workflow.ipynb` - Full clinical pipeline
- `examples/pinn_tutorial.ipynb` - PINN parameter estimation
- `examples/bayesian_tutorial.ipynb` - Bayesian inference
- `examples/virtual_drug_trials_tutorial.ipynb` - Drug testing
- `examples/hardware_export_tutorial.ipynb` - Verilog-A/SPICE

Run with:
```bash
jupyter notebook examples/
```

---

## Next Steps

1. **Deep Dive:** Read the [User Guide](user_guide.md)
2. **API Reference:** See [api_reference.rst](api_reference.rst)
3. **Examples:** Explore `examples/` directory
4. **Contribute:** See [CONTRIBUTING.md](../CONTRIBUTING.md)

---

## Common Issues

**Slow simulation?**
- Reduce `n_segments` or increase `dt`
- Use smaller `duration`

**Memory error?**
- Use fewer segments
- Don't store all time points (use `record_every`)

**Unexpected results?**
- Check parameter ranges (use `twin.print_parameters()`)
- Verify profile applied correctly
- Try with default parameters first

---

## Getting Help

- **Documentation:** https://ens-gi-digital-twin.readthedocs.io
- **GitHub Issues:** https://github.com/yourusername/ens-gi-digital-twin/issues
- **Email:** contact@example.com
