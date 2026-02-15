# Patient Data Directory

## ⚠️ IMPORTANT: Current Data is Synthetic

**All patient data files in this directory (P001-P003) are synthetically generated for testing and demonstration purposes only.**

These files are **NOT** real clinical recordings. They were created using the ENS-GI Digital Twin simulation engine to generate physiologically plausible signals that match expected biomarker ranges for different IBS phenotypes.

## Current Files

### Synthetic Patient Cohort

| Patient ID | IBS Phenotype | Files |
|-----------|---------------|-------|
| P001 | IBS-D (Diarrhea-predominant) | `P001_egg.csv`, `P001_hrm.csv` |
| P002 | Healthy Control | `P002_egg.csv`, `P002_hrm.csv` |
| P003 | IBS-C (Constipation-predominant) | `P003_egg.csv`, `P003_hrm.csv` |

### File Format

**EGG Files (`*_egg.csv`):**
```
time_s,ch1_mV,ch2_mV,ch3_mV,ch4_mV
0.000,-45.2,-45.1,-45.3,-45.0
0.050,-45.1,-45.2,-45.1,-45.1
...
```

**HRM Files (`*_hrm.csv`):**
```
time_s,sensor1_mmHg,sensor2_mmHg,sensor3_mmHg,...,sensor8_mmHg
0.000,12.5,13.2,14.1,15.0,14.8,13.9,13.2,12.7
0.050,12.6,13.3,14.2,15.1,14.9,14.0,13.3,12.8
...
```

## Synthetic Data Generation Method

The synthetic patient data was generated using the following process:

1. **Simulation Configuration:**
   - ENS network: 50-100 segments (oral to anal gradient)
   - ICC slow wave: 3 cycles/min baseline frequency
   - Simulation duration: 300-600 seconds
   - Time step: 0.05 ms

2. **IBS Phenotype Modeling:**
   - **IBS-D:** Increased ICC frequency (ω = +30%), reduced Na⁺ conductance (g_Na = -20%)
   - **IBS-C:** Decreased ICC frequency (ω = -25%), reduced Ca²⁺ conductance (g_Ca = -30%)
   - **Healthy:** Baseline parameters from literature (Thomas-Bornstein 2007)

3. **Signal Processing:**
   - EGG: Voltage traces from 4 surface electrodes (spatial sampling)
   - HRM: Pressure conversion via contractility model (V → force → pressure)
   - Noise: Added physiological noise (SNR ~30 dB)
   - Resampling: Downsampled to 20 Hz (clinical EGG standard)

4. **Validation Against Literature:**
   - Slow wave frequency: 2.5-3.5 cycles/min ✓
   - EGG amplitude: 0.1-0.5 mV (surface electrodes) ✓
   - HRM pressure: 10-50 mmHg (duodenal contractions) ✓

## Limitations of Synthetic Data

⚠️ **Current limitations that prevent true clinical validation:**

1. **No Real Patient Variability:** Synthetic data does not capture:
   - Inter-patient anatomical differences
   - Dietary influences on motility
   - Medication effects
   - Comorbidities (e.g., diabetes, thyroid disorders)
   - Psychological stress effects

2. **Simplified Noise Model:** Real clinical recordings contain:
   - Motion artifacts (breathing, heartbeat)
   - Electrode impedance variations
   - EMG cross-contamination (skeletal muscle)
   - Power line interference (50/60 Hz)

3. **Idealized Spatial Sampling:** Real EGG/HRM has:
   - Variable electrode placement across patients
   - Non-uniform sensor spacing
   - Depth-dependent signal attenuation

4. **No Long-Term Dynamics:** Synthetic data is stationary, but real GI signals show:
   - Circadian rhythms
   - Postprandial responses
   - Migrating motor complex (MMC) patterns

## Plan for Real Clinical Data Integration

### Phase 1: Data Acquisition (Post-Program Completion)

**Target Sources:**
1. **Open-Source Datasets:**
   - PhysioNet GI Database (if available)
   - Published study datasets with ethical approval
   - Collaborating research groups

2. **Data Requirements:**
   - **Minimum Sample Size:** 30 patients (10 IBS-D, 10 IBS-C, 10 healthy)
   - **Recording Duration:** ≥30 minutes per patient
   - **Modalities:** Simultaneous EGG + HRM preferred
   - **Annotations:** IBS subtype, symptom severity scores, medications

3. **Ethical Approval:**
   - Verify IRB approval for data sharing
   - Confirm patient consent for research use
   - Ensure HIPAA/GDPR compliance (anonymization)

### Phase 2: Data Validation

**When real data is acquired, validate that:**
1. Synthetic data biomarkers match real cohort statistics
2. Parameter estimation accuracy on real vs synthetic data
3. IBS classification performance (PINN/Bayesian frameworks)

### Phase 3: Model Refinement

**Use real data to:**
1. Refine prior distributions for Bayesian inference
2. Retrain PINN on real patient cohort
3. Identify missing biological mechanisms
4. Improve noise models for realistic artifacts

## How to Add New Patient Data

### For Researchers with Real Clinical Data:

1. **Format your data** to match the CSV structure above:
   ```python
   # Example: Convert your EGG recording
   import pandas as pd

   egg_data = pd.DataFrame({
       'time_s': time_vector,
       'ch1_mV': channel1_voltages,
       'ch2_mV': channel2_voltages,
       # ... add all channels
   })

   egg_data.to_csv('patient_data/P004_egg.csv', index=False)
   ```

2. **Load into the digital twin:**
   ```python
   from patient_data_loader import PatientDataLoader

   loader = PatientDataLoader('patient_data/P004_egg.csv',
                              'patient_data/P004_hrm.csv')

   egg_signal, hrm_signal = loader.load()
   ```

3. **Run parameter estimation:**
   ```python
   from ens_gi_pinn import PINNEstimator
   from ens_gi_bayesian import BayesianEstimator

   # PINN approach
   pinn = PINNEstimator()
   params_pinn = pinn.estimate_parameters(egg_signal)

   # Bayesian approach
   bayes = BayesianEstimator()
   params_bayes = bayes.estimate_parameters_bayesian(egg_signal)
   ```

4. **Validate predictions:**
   - Compare estimated parameters to literature ranges
   - Check if IBS subtype is correctly classified
   - Assess biomarker predictions vs actual measurements

## Citation for Synthetic Data

If using this synthetic data for publications or presentations, please cite:

```bibtex
@software{ens_gi_digital_twin_2026,
  title = {ENS-GI Digital Twin: Synthetic Patient Data},
  author = {{ENS-GI Development Team}},
  year = {2026},
  note = {Synthetically generated test data for IBS parameter estimation. Not for clinical use.},
  url = {https://github.com/yourusername/ens-gi-digital-twin}
}
```

⚠️ **DO NOT use synthetic data results for clinical decision-making or medical publications without clearly labeling them as simulated data.**

## Roadmap

- [x] Generate synthetic IBS-D, IBS-C, Healthy cohorts (P001-P003)
- [ ] Document data generation method ✅ (this file)
- [ ] Acquire real patient data from open-source datasets
- [ ] Validate parameter estimation on real cohort
- [ ] Publish validation study comparing synthetic vs real data
- [ ] Expand to ≥100 patients for robust clinical validation

---

**Last Updated:** 2026-02-15
**Status:** Synthetic data only - real data integration planned
**Contact:** [Add contact info for data sharing inquiries]
