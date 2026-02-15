# Installation Guide

## Requirements

### System Requirements

- **Python:** 3.8 or higher
- **OS:** Windows, Linux, macOS
- **RAM:** 4 GB minimum, 8 GB recommended
- **Disk Space:** 500 MB

### Python Packages

**Required:**
- NumPy >= 1.20
- SciPy >= 1.7
- Matplotlib >= 3.4

**Optional (for AI features):**
- TensorFlow >= 2.10 (for PINN)
- PyMC3 >= 3.11 (for Bayesian inference)
- ArviZ >= 0.11 (for Bayesian diagnostics)

**Optional (for visualization):**
- Seaborn >= 0.11
- Plotly >= 5.0

**Optional (for hardware export):**
- ngspice (for SPICE simulation)
- Cadence Spectre (for Verilog-A)

---

## Installation Methods

### Method 1: pip install (Recommended)

```bash
# Clone repository
git clone https://github.com/yourusername/ens-gi-digital-twin.git
cd ens-gi-digital-twin

# Install dependencies
pip install -r requirements.txt

# Install package
python setup.py install
```

### Method 2: Development install

For development (editable install):

```bash
pip install -e .
```

### Method 3: From source

```bash
python setup.py build
python setup.py install
```

---

## Optional Dependencies

### Install PINN framework:

```bash
pip install tensorflow
```

### Install Bayesian framework:

```bash
pip install pymc3 arviz
```

### Install all optional dependencies:

```bash
pip install tensorflow pymc3 arviz seaborn plotly
```

---

## Verification

Test installation:

```python
import ens_gi_core
from ens_gi_core import ENSGIDigitalTwin

# Create twin
twin = ENSGIDigitalTwin(n_segments=5)
print(f"✓ Installation successful! Version: {ens_gi_core.__version__}")

# Run quick test
result = twin.run(100, dt=0.1, verbose=False)
print(f"✓ Simulation working! {len(result['time'])} time points")
```

Expected output:
```
✓ Installation successful! Version: 0.3.0
✓ Simulation working! 1000 time points
```

---

## Troubleshooting

### Import errors

**Error:** `ModuleNotFoundError: No module named 'ens_gi_core'`

**Solution:**
```bash
# Ensure installation directory is in Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/ens-gi-digital-twin"
```

### TensorFlow issues

**Error:** TensorFlow not compatible with Python 3.12

**Solution:** Use Python 3.10 or 3.11:
```bash
conda create -n ens_gi python=3.10
conda activate ens_gi
pip install -r requirements.txt
```

### PyMC3 issues

**Error:** `AttributeError: module 'numpy' has no attribute 'float'`

**Solution:** PyMC3 requires NumPy < 1.24:
```bash
pip install "numpy<1.24" pymc3
```

### Memory errors

**Error:** `MemoryError` during large simulations

**Solution:** Reduce network size or increase swap:
```python
twin = ENSGIDigitalTwin(n_segments=10)  # Smaller network
```

---

## Platform-Specific Notes

### Windows

- Use Anaconda for easier dependency management
- Visual C++ redistributable may be required for some packages

### Linux

- Ensure build-essential installed: `sudo apt-get install build-essential`
- For GPU acceleration: Install CUDA toolkit

### macOS

- Xcode command line tools required: `xcode-select --install`
- M1/M2 Macs: Use conda-forge for native ARM packages

---

## Next Steps

After installation:

1. **Quick Start:** See [quickstart.md](quickstart.md)
2. **Tutorials:** Explore Jupyter notebooks in `examples/`
3. **API Reference:** [api_reference.rst](api_reference.rst)
4. **Tests:** Run `pytest tests/` to verify

---

## Uninstallation

```bash
pip uninstall ens-gi-digital-twin
```

Or manually delete installation directory.
