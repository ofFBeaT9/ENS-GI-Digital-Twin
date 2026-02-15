# ENS-GI Digital Twin - Utility Scripts

This directory contains ad-hoc testing, validation, and profiling scripts for the ENS-GI Digital Twin project.

## Testing Scripts

### Quick Verification
```bash
python scripts/quick_test.py
```
Fast 5-test verification of core functionality (ICC frequency, IBS profiles, SPICE export, Verilog-A, biomarkers).

### Manual Test Suite
```bash
python scripts/manual_test.py
```
Comprehensive 9-test suite covering all major components including drug library and clinical workflow.

### Final Verification
```bash
python scripts/final_verification.py
```
Production readiness check with 8 comprehensive tests including realistic clinical parameters, parameter sweeps, and reproducibility checks.

### Installation Verification
```bash
python scripts/verify_installation.py
```
Post-install verification to ensure all dependencies are correctly installed.

## Validation Scripts

### SPICE Hardware Validation
```bash
python scripts/validate_spice.py
```
Automated SPICE netlist validation comparing Python simulation with ngspice output. Requires ngspice to be installed.

```bash
python scripts/validate_spice_netlist.py
```
Alternative SPICE validation script for netlist execution.

## Profiling Scripts

### Performance Profiling
```bash
python scripts/profile_performance.py
```
Profile simulation performance, identify bottlenecks, and benchmark different configurations.

## Usage Notes

- All scripts assume the package is installed in editable mode: `pip install -e .`
- Scripts use the new import structure: `from ens_gi_digital import ...`
- SPICE validation requires ngspice: `conda install -c conda-forge ngspice`

## Quick Test Workflow

For rapid verification after changes:
```bash
# 1. Quick functional test
python scripts/quick_test.py

# 2. If passing, run full official tests
pytest tests/ -v

# 3. Before release, run final verification
python scripts/final_verification.py
```
