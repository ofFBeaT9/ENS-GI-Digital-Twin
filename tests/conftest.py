"""
Pytest configuration and fixtures for reproducible testing.

This file provides centralized test configuration including:
- Random seed management for reproducibility
- Shared fixtures
- Test isolation
"""
import pytest
import numpy as np


@pytest.fixture(scope="session", autouse=True)
def set_random_seeds():
    """Set random seeds at the start of test session for reproducibility.

    This fixture runs automatically once per test session and sets global
    random seeds for NumPy and TensorFlow to ensure deterministic behavior.
    """
    np.random.seed(42)
    try:
        import tensorflow as tf
        tf.random.set_seed(42)
    except ImportError:
        pass  # TensorFlow not available, skip

    yield

    # Cleanup (if needed)


@pytest.fixture(scope="function")
def reset_seeds():
    """Reset random seeds before each test function for isolation.

    Use this fixture in tests that need fresh random state:

    Example:
        def test_something(reset_seeds):
            # Random numbers will be deterministic
            data = np.random.randn(100)
    """
    np.random.seed(42)
    try:
        import tensorflow as tf
        tf.random.set_seed(42)
    except ImportError:
        pass

    yield

    # Cleanup after test
