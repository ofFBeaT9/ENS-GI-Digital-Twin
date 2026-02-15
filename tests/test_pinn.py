"""
Unit tests for PINN (Physics-Informed Neural Network) framework
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ens_gi_digital.core import ENSGIDigitalTwin
try:
    from ens_gi_digital.pinn import PINNEstimator, PINNConfig, build_mlp_network, build_resnet_network
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")
class TestPINNArchitecture:
    """Test neural network architectures."""

    def test_mlp_network_creation(self):
        """Test MLP network builds correctly."""
        model = build_mlp_network(
            input_dim=110,
            output_dim=5,
            hidden_dims=[64, 32],
            activation='tanh'
        )

        assert model is not None
        assert len(model.layers) > 0
        assert model.input_shape == (None, 110)
        assert model.output_shape == (None, 5)

    def test_resnet_network_creation(self):
        """Test ResNet network builds correctly."""
        model = build_resnet_network(
            input_dim=110,
            output_dim=5,
            hidden_dims=[64, 64, 32, 32],
            activation='tanh'
        )

        assert model is not None
        assert len(model.layers) > 0
        assert model.input_shape == (None, 110)
        assert model.output_shape == (None, 5)

    def test_model_forward_pass(self):
        """Test that model can perform forward pass."""
        model = build_mlp_network(110, 5, [64, 32])

        # Create dummy input
        x = np.random.randn(10, 110).astype(np.float32)

        # Forward pass
        output = model(x, training=False)

        assert output.shape == (10, 5)
        assert not np.any(np.isnan(output.numpy()))


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")
class TestPINNEstimator:
    """Test PINN estimator functionality."""

    @pytest.fixture
    def digital_twin(self):
        """Create a digital twin for testing."""
        return ENSGIDigitalTwin(n_segments=10)

    @pytest.fixture
    def pinn_estimator(self, digital_twin):
        """Create a PINN estimator for testing."""
        config = PINNConfig(
            architecture='mlp',
            hidden_dims=[32, 16],
            learning_rate=1e-3,
        )
        return PINNEstimator(
            digital_twin=digital_twin,
            config=config,
            parameter_names=['g_Na', 'g_K', 'omega']
        )

    def test_pinn_initialization(self, pinn_estimator):
        """Test PINN estimator initializes correctly."""
        assert pinn_estimator.model is not None
        assert pinn_estimator.n_params == 3
        assert 'g_Na' in pinn_estimator.parameter_names

    def test_parameter_normalization(self, pinn_estimator):
        """Test parameter normalization and denormalization."""
        params = np.array([120.0, 36.0, 0.005])  # g_Na, g_K, omega

        # Normalize
        normalized = pinn_estimator._normalize_parameters(params)
        assert np.all(normalized >= 0)
        assert np.all(normalized <= 1)

        # Denormalize
        denormalized = pinn_estimator._denormalize_parameters(normalized)
        np.testing.assert_array_almost_equal(params, denormalized, decimal=5)

    def test_feature_extraction(self, pinn_estimator):
        """Test feature extraction from signals."""
        # Create synthetic signals
        T, N = 1000, 10
        voltages = np.random.randn(T, N) * 10 - 50
        forces = np.random.rand(T, N) * 0.5
        calcium = np.random.rand(T, N) * 0.01

        # Extract features
        features = pinn_estimator._extract_features_from_signal(
            voltages, forces, calcium
        )

        assert len(features) == 110  # Expected feature dimension
        assert not np.any(np.isnan(features))

    def test_synthetic_data_generation(self, pinn_estimator):
        """Test synthetic dataset generation."""
        features, params = pinn_estimator.generate_synthetic_dataset(
            n_samples=10,  # Small for testing
            duration=500.0,
            dt=0.1,
            noise_level=0.02
        )

        assert features.shape == (10, 110)
        assert params.shape == (10, 3)
        assert not np.any(np.isnan(features))
        assert not np.any(np.isnan(params))

        # Check normalization
        assert np.all(params >= 0)
        assert np.all(params <= 1)

    def test_training_step(self, pinn_estimator):
        """Test single training step."""
        # Create dummy data
        features = tf.constant(np.random.randn(16, 110), dtype=tf.float32)
        params = tf.constant(np.random.rand(16, 3), dtype=tf.float32)

        # Single training step
        loss, data_loss, physics_loss = pinn_estimator._train_step(features, params)

        assert loss.numpy() > 0
        assert data_loss.numpy() >= 0
        assert physics_loss.numpy() >= 0

    def test_train_on_small_dataset(self, pinn_estimator):
        """Test training on a very small dataset."""
        # Generate tiny dataset
        features, params = pinn_estimator.generate_synthetic_dataset(
            n_samples=20,
            duration=300.0
        )

        # Train for just a few epochs
        history = pinn_estimator.train(
            features=features,
            parameters=params,
            epochs=10,
            verbose=0,
            generate_data=False
        )

        assert len(history['train_loss']) == 10
        assert len(history['val_loss']) == 10

        # Loss should be finite
        assert all(np.isfinite(loss) for loss in history['train_loss'])


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")
class TestPINNParameterRecovery:
    """Test PINN ability to recover known parameters."""

    @pytest.fixture
    def trained_pinn(self):
        """Create and train a PINN on synthetic data."""
        twin = ENSGIDigitalTwin(n_segments=10)
        pinn = PINNEstimator(
            digital_twin=twin,
            config=PINNConfig(hidden_dims=[32, 16], lambda_physics=0.05),
            parameter_names=['g_Na', 'g_K', 'omega']
        )

        # Train on small dataset
        pinn.train(epochs=50, n_synthetic_samples=50, verbose=0)
        return pinn

    def test_parameter_estimation(self, trained_pinn):
        """Test parameter estimation from simulated data."""
        # Create test twin with known parameters
        test_twin = ENSGIDigitalTwin(n_segments=10)

        # Set known parameters
        true_g_Na = 130.0
        for neuron in test_twin.network.neurons:
            neuron.params.g_Na = true_g_Na

        # Run simulation
        result = test_twin.run(1000, dt=0.1, I_stim={3: 10.0}, verbose=False)

        # Estimate parameters
        estimates, uncertainties = trained_pinn.estimate_parameters(
            voltages=result['voltages'],
            forces=result['force'],
            calcium=result['calcium'],
            n_bootstrap=10  # Small for testing
        )

        # Check that estimates are reasonable
        assert 'g_Na' in estimates
        assert estimates['g_Na'] > 0
        assert uncertainties['g_Na'] > 0

        # Note: With limited training, we don't expect high accuracy
        # Just check that the estimate is in a reasonable range
        assert 50 < estimates['g_Na'] < 250

    def test_model_save_load(self, trained_pinn, tmp_path):
        """Test saving and loading PINN model."""
        # Save model
        save_path = str(tmp_path / "test_pinn_model")
        trained_pinn.save(save_path)

        # Check files exist
        assert os.path.exists(save_path)
        assert os.path.exists(save_path + '_config.json')

        # Load model
        twin = ENSGIDigitalTwin(n_segments=10)
        loaded_pinn = PINNEstimator.load(save_path, twin)

        assert loaded_pinn is not None
        assert loaded_pinn.parameter_names == trained_pinn.parameter_names


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")
class TestPINNValidation:
    """Test PINN validation metrics."""

    def test_validation_metrics(self):
        """Test validation on test set."""
        twin = ENSGIDigitalTwin(n_segments=8)
        pinn = PINNEstimator(twin, parameter_names=['g_Na', 'g_K'])

        # Create small synthetic test set
        features, true_params = pinn.generate_synthetic_dataset(n_samples=10)

        # Train minimally
        pinn.train(features=features, parameters=true_params, epochs=20, verbose=0)

        # Validate
        results = pinn.validate_on_test_set(features, true_params)

        assert 'g_Na' in results
        assert 'g_K' in results
        assert 'mae' in results['g_Na']
        assert 'rmse' in results['g_Na']
        assert 'mape' in results['g_Na']

        # Errors should be positive
        assert results['g_Na']['mae'] >= 0
        assert results['g_K']['rmse'] >= 0


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
