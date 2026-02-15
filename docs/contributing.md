# Contributing to ENS-GI Digital Twin

Thank you for your interest in contributing to the ENS-GI Digital Twin project! This document provides guidelines for contributing code, documentation, and other improvements.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Testing Guidelines](#testing-guidelines)
5. [Documentation Standards](#documentation-standards)
6. [Code Style](#code-style)
7. [Submitting Changes](#submitting-changes)
8. [Issue Guidelines](#issue-guidelines)

---

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. By participating in this project, you agree to:

- Use welcoming and inclusive language
- Respect differing viewpoints and experiences
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

---

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/ens-gi-digital-twin.git
cd ens-gi-digital-twin
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (including dev dependencies)
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Install Development Tools

```bash
# Testing
pip install pytest pytest-cov pytest-mock pytest-benchmark

# Code quality
pip install black flake8 mypy

# Documentation
pip install sphinx sphinx-rtd-theme

# Optional: AI frameworks
pip install tensorflow pymc3 arviz
```

---

## Development Workflow

### Branch Naming Convention

- **Feature:** `feature/description-of-feature`
- **Bug fix:** `bugfix/description-of-bug`
- **Documentation:** `docs/description-of-change`
- **Testing:** `test/description-of-test`

Example:
```bash
git checkout -b feature/2d-tissue-simulation
```

### Commit Messages

Follow conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `style`: Code style changes (formatting, etc.)

**Example:**
```
feat(pinn): add ResNet architecture option

Implemented residual network architecture for PINN estimator to
improve gradient flow and convergence on deep networks.

Closes #42
```

---

## Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_ion_channels.py

# Run with coverage
pytest --cov=. --cov-report=html

# Run only fast tests (skip slow integration tests)
pytest -m "not slow"
```

### Writing Tests

All new features should include tests. Place tests in `tests/` directory:

```python
# tests/test_new_feature.py
import pytest
from ens_gi_core import ENSGIDigitalTwin

def test_new_feature():
    """Test description."""
    twin = ENSGIDigitalTwin(n_segments=5)
    # Test logic here
    assert expected_result == actual_result

@pytest.mark.slow
def test_integration():
    """Integration test marked as slow."""
    # Longer test logic
    pass
```

### Test Coverage Requirements

- New features must have >80% test coverage
- Bug fixes should include regression tests
- Integration tests for complex features

---

## Documentation Standards

### Code Documentation

Use **NumPy-style docstrings**:

```python
def estimate_parameters(self, voltages, forces, calcium, n_bootstrap=100):
    """Estimate parameters from clinical data using PINN.

    Parameters
    ----------
    voltages : np.ndarray, shape (n_timesteps, n_neurons)
        Recorded voltage traces from EGG/intracellular recordings.
    forces : np.ndarray, shape (n_timesteps, n_segments)
        Smooth muscle contractile force measurements.
    calcium : np.ndarray, shape (n_timesteps, n_neurons)
        Intracellular calcium concentrations (optional).
    n_bootstrap : int, default=100
        Number of bootstrap samples for uncertainty estimation.

    Returns
    -------
    estimates : dict
        Dictionary with keys as parameter names, values as dicts with:
        - 'mean': float, estimated parameter value
        - 'std': float, standard deviation from bootstrap
        - 'ci_95': tuple, 95% confidence interval

    Examples
    --------
    >>> from ens_gi_pinn import PINNEstimator, PINNConfig
    >>> pinn = PINNEstimator(twin, PINNConfig())
    >>> estimates = pinn.estimate_parameters(voltages, forces, calcium)
    >>> print(estimates['g_Na']['mean'])
    120.3
    """
    # Implementation
    pass
```

### Tutorial Notebooks

When contributing tutorials:

1. **Clear learning objectives** at the start
2. **Step-by-step explanations** with code
3. **Visualizations** to illustrate concepts
4. **Summary** of key points at the end
5. **References** to relevant papers/docs

---

## Code Style

### Python Style Guide

We follow **PEP 8** with some modifications:

- **Line length:** 100 characters (not 79)
- **Indentation:** 4 spaces
- **Quotes:** Prefer single quotes `'` for strings
- **Imports:** Group by stdlib, third-party, local

### Formatting

Use **Black** for automatic formatting:

```bash
black ens_gi_core.py
```

### Linting

Use **Flake8** for linting:

```bash
flake8 ens_gi_core.py --max-line-length=100
```

### Type Hints

Use **type hints** for function signatures:

```python
from typing import Dict, Optional, Tuple
import numpy as np

def extract_biomarkers(self) -> Dict[str, float]:
    """Extract clinical biomarkers."""
    biomarkers: Dict[str, float] = {}
    # Implementation
    return biomarkers
```

Run **mypy** for type checking:

```bash
mypy ens_gi_core.py --ignore-missing-imports
```

---

## Submitting Changes

### Pull Request Process

1. **Update your branch**
   ```bash
   git checkout main
   git pull upstream main
   git checkout your-branch
   git rebase main
   ```

2. **Run tests and linting**
   ```bash
   pytest
   black .
   flake8 . --max-line-length=100
   ```

3. **Push to your fork**
   ```bash
   git push origin your-branch
   ```

4. **Create pull request on GitHub**
   - Use descriptive title
   - Reference related issues
   - Describe changes in detail
   - Include screenshots for UI changes

### Pull Request Template

```markdown
## Description
Brief description of changes.

## Related Issues
Fixes #123

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Added unit tests
- [ ] Added integration tests
- [ ] All tests passing
- [ ] Test coverage >80%

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
- [ ] Commit messages follow convention
```

### Review Process

- At least one maintainer must approve
- All CI checks must pass
- Resolve all review comments
- Squash commits before merging (if requested)

---

## Issue Guidelines

### Reporting Bugs

Use the bug report template:

```markdown
**Bug Description**
Clear description of the bug.

**To Reproduce**
Steps to reproduce:
1. Create twin with ...
2. Run simulation ...
3. See error

**Expected Behavior**
What should happen.

**Actual Behavior**
What actually happens.

**Environment**
- OS: Windows 10 / macOS 13 / Ubuntu 22.04
- Python: 3.10.5
- ENS-GI Version: 0.3.0

**Additional Context**
Error messages, screenshots, etc.
```

### Feature Requests

Use the feature request template:

```markdown
**Problem Statement**
What problem does this solve?

**Proposed Solution**
How should it work?

**Alternatives Considered**
Other approaches you've thought about.

**Additional Context**
References, examples, mockups.
```

---

## Contribution Areas

We especially welcome contributions in:

### High Priority
- **Clinical validation** - Access to real patient data
- **Hardware testing** - FPGA/ASIC implementation
- **2D tissue simulation** - Circumferential propagation
- **Performance optimization** - Faster simulations
- **Drug library expansion** - Additional GI drugs

### Documentation
- Tutorial notebooks for advanced topics
- API documentation improvements
- Translation to other languages
- Video tutorials

### Testing
- Unit test coverage improvements
- Integration tests
- Performance benchmarks
- Hardware validation tests

### Features
- New ion channel models
- Alternative ICC pacemaker models
- Multi-scale tissue modeling
- Real-time visualization dashboard

---

## Development Roadmap

See [IMPLEMENTATION_TODO.md](../IMPLEMENTATION_TODO.md) for detailed roadmap.

**Phase 1 (Year 1):** Mathematical Engine ✅ 95% Complete
**Phase 2 (Year 2):** Hardware Realization ✅ 90% Complete
**Phase 3 (Year 3):** Clinical Digital Twin ✅ 85% Complete

Current priorities:
1. Clinical data integration
2. 2D tissue simulation
3. Hardware fabrication validation
4. Extended drug library

---

## Recognition

Contributors will be:
- Listed in `AUTHORS.md`
- Acknowledged in research papers (for significant contributions)
- Invited to co-author publications (for major features)

---

## Questions?

- **Documentation:** https://ens-gi-digital-twin.readthedocs.io
- **GitHub Issues:** https://github.com/yourusername/ens-gi-digital-twin/issues
- **Email:** contact@example.com
- **Slack:** [Join our workspace](#)

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to advancing computational gastroenterology!**
