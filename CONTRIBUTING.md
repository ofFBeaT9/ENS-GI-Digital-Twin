# Contributing to ENS-GI Digital Twin

Thank you for your interest in contributing to the ENS-GI Digital Twin project! This document provides guidelines and instructions for contributors.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Coding Standards](#coding-standards)
5. [Testing](#testing)
6. [Documentation](#documentation)
7. [Pull Request Process](#pull-request-process)
8. [Areas for Contribution](#areas-for-contribution)

---

## Code of Conduct

This project adheres to a code of professional and respectful conduct. We expect all contributors to:

- Be respectful and constructive in discussions
- Welcome newcomers and help them get started
- Focus on what is best for the community and the project
- Show empathy towards other community members

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git for version control
- Basic understanding of computational neuroscience or biophysics (helpful but not required)

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/ens-gi-digital-twin.git
cd ens-gi-digital-twin

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8 mypy

# Install in editable mode
pip install -e .

# Run tests to verify installation
pytest tests/ -v
```

---

## Development Workflow

### 1. Create a Branch

```bash
# Create a new branch for your feature/fix
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 2. Make Changes

- Write clean, well-documented code
- Follow the coding standards (see below)
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Check code style
black ens_gi_*.py tests/
flake8 ens_gi_*.py tests/

# Type checking (optional)
mypy ens_gi_*.py
```

### 4. Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "Add feature: Brief description

Detailed explanation of what changed and why.

Fixes #123"
```

### 5. Push and Create Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create pull request on GitHub
# Include:
# - Description of changes
# - Issue number (if applicable)
# - Test results
# - Breaking changes (if any)
```

---

## Coding Standards

### Python Style

- Follow **PEP 8** style guide
- Use **Black** for code formatting (line length: 100)
- Use **type hints** where appropriate
- Write **docstrings** for all public functions/classes (Google style)

Example:
```python
def simulate_neuron(params: MembraneParams, duration: float) -> Dict[str, np.ndarray]:
    """Simulate single neuron dynamics.

    Args:
        params: Membrane biophysical parameters
        duration: Simulation duration in milliseconds

    Returns:
        Dictionary containing:
            - 'time': Time array
            - 'voltage': Membrane voltage array
            - 'spikes': Spike times

    Raises:
        ValueError: If duration is negative
    """
    pass
```

### Code Organization

- **One class per file** for major components
- **Group related functions** in modules
- **Avoid circular imports**
- **Keep functions small** (<50 lines when possible)

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `ENSNeuron`, `PINNEstimator`)
- **Functions**: `snake_case` (e.g., `compute_currents`, `extract_biomarkers`)
- **Constants**: `UPPER_CASE` (e.g., `DEFAULT_TIMESTEP`, `MAX_ITERATIONS`)
- **Private members**: `_leading_underscore` (e.g., `_compute_internal_state`)

### Comments

- **Explain WHY, not WHAT**: Code should be self-explanatory; comments explain reasoning
- **Reference papers**: When implementing published models, cite the paper
- **TODO comments**: Mark incomplete work with `# TODO: description`

Example:
```python
# Use FitzHugh-Nagumo instead of Corrias-Buist ICC model
# as it provides similar dynamics with lower computational cost
# Reference: Hodgkin & Huxley 1952, J Physiol
```

---

## Testing

### Test Structure

```
tests/
├── test_core.py           # Core simulation tests
├── test_pinn.py           # PINN framework tests
├── test_bayesian.py       # Bayesian inference tests
├── test_drug_library.py   # Drug trial tests
└── test_integration.py    # End-to-end integration tests
```

### Writing Tests

- **Use pytest** framework
- **One test per function** when practical
- **Test edge cases** and error conditions
- **Use fixtures** for common setups

Example:
```python
import pytest
from ens_gi_core import ENSNeuron

@pytest.fixture
def neuron():
    """Create a neuron for testing."""
    return ENSNeuron()

def test_action_potential_generation(neuron):
    """Test that neuron generates action potentials with strong stimulus."""
    for _ in range(100):
        neuron.step(dt=0.05, I_ext=15.0)

    assert len(neuron.spike_times) > 0, "Neuron should spike with strong stimulus"
```

### Test Coverage

- **Target**: >80% code coverage
- **Focus**: Critical paths, edge cases, error handling
- **Run coverage**: `pytest --cov=. --cov-report=html`

---

## Documentation

### Code Documentation

- **All public functions/classes** must have docstrings
- **Use Google-style** docstrings
- **Include examples** in docstrings when helpful

### README Updates

- Update `README.md` for new features
- Add examples to quick start section
- Update installation instructions if dependencies change

### Tutorials

- Create Jupyter notebooks for new workflows
- Place in `examples/` directory
- Include clear explanations and visualizations

---

## Pull Request Process

### Before Submitting

- [ ] All tests pass (`pytest tests/ -v`)
- [ ] Code is formatted (`black ens_gi_*.py`)
- [ ] No linting errors (`flake8 ens_gi_*.py`)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated

### PR Description Template

```markdown
## Description
Brief summary of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] All tests pass
- [ ] Added new tests
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed code
- [ ] Commented complex sections
- [ ] Documentation updated
- [ ] No breaking changes (or documented)

## Related Issues
Fixes #123
```

### Review Process

1. **Automated checks**: CI/CD runs tests automatically
2. **Code review**: At least one maintainer reviews
3. **Feedback**: Address review comments
4. **Approval**: Maintainer approves PR
5. **Merge**: Squash and merge to main

---

## Areas for Contribution

### High Priority

- **Testing**: Increase test coverage to >80%
- **Documentation**: Create tutorial Jupyter notebooks
- **Validation**: Compare results to published experimental data
- **Performance**: Optimize simulation speed (Numba, GPU)

### Phase 2 (Hardware)

- **Verilog-A Library**: Complete standard cell library for ion channels
- **SPICE Netlist**: Make generated netlists runnable in ngspice
- **2D Tissue**: Implement 2D network topology

### Phase 3 (Clinical AI)

- **PINN Enhancement**: Improve physics loss with full ODE residuals
- **Bayesian**: Integrate full time series in likelihood function
- **Clinical Data**: Integrate with real patient datasets

### General

- **Bug Fixes**: Address issues in GitHub tracker
- **Examples**: Add more usage examples
- **Visualization**: Improve plotting and animation tools

---

## Questions?

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and ideas
- **Email**: [your.email@example.com]

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to ENS-GI Digital Twin!**

Your contributions help advance computational medicine and improve patient care.
