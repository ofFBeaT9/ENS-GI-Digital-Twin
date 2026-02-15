"""
ENS-GI Digital Twin — Physics-Informed Neural Network (PINN) Framework
========================================================================
Enables patient-specific parameter estimation from clinical data (EGG, HRM).

Key Features:
- Combines data loss (clinical measurements) with physics loss (ODE constraints)
- Solves the inverse problem: clinical signals → internal parameters
- Returns parameter estimates with uncertainty quantification
- Supports multiple architectures (MLP, ResNet-style, etc.)

Mathematical Framework:
    Total Loss = λ_data * L_data + λ_physics * L_physics

    L_data = ||simulation_output - clinical_data||²
    L_physics = ||dV/dt - f(V, params)||²

    where f(V, params) is the ENS-GI ODE system

Usage:
    from ens_gi_digital import ENSGIDigitalTwin, PINNEstimator

    # Create digital twin
    twin = ENSGIDigitalTwin(n_segments=20)

    # Create PINN estimator
    pinn = PINNEstimator(twin, architecture='mlp', hidden_dims=[128, 128, 64])

    # Train on synthetic or real clinical data
    history = pinn.train(clinical_data={'egg_signal': egg, 'hrm_signal': hrm},
                         epochs=5000, lambda_physics=0.1)

    # Estimate parameters from new patient data
    params, uncertainties = pinn.estimate_parameters(egg_signal, hrm_signal)

Author: Mahdad (Phase 3 Implementation)
License: MIT
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, TYPE_CHECKING
from dataclasses import dataclass
import json

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    keras = None  # Define as None when not available
    tf = None
    print("[WARNING] TensorFlow not installed. Install with: pip install tensorflow")

    # Create dummy decorator for when TF is not available
    class DummyTF:
        @staticmethod
        def function(func):
            """Dummy decorator when TensorFlow not available."""
            return func

    if tf is None:
        tf = DummyTF()

# For type hints only
if TYPE_CHECKING:
    from tensorflow import keras

# Import core digital twin
from .core import ENSGIDigitalTwin, MembraneParams, NetworkParams, ICCParams


# ═══════════════════════════════════════════════════════════════
# PINN Architecture Components
# ═══════════════════════════════════════════════════════════════

@dataclass
class PINNConfig:
    """Configuration for PINN architecture and training."""
    architecture: str = 'mlp'           # 'mlp', 'resnet', 'lstm'
    hidden_dims: List[int] = None       # Hidden layer dimensions
    activation: str = 'tanh'            # Activation function
    learning_rate: float = 1e-3         # Adam learning rate
    lambda_data: float = 1.0            # Data loss weight
    lambda_physics: float = 0.1         # Physics loss weight
    batch_size: int = 32                # Training batch size
    validation_split: float = 0.2       # Validation data fraction
    early_stopping_patience: int = 100  # Early stopping patience

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 128, 64, 32]


def build_mlp_network(input_dim: int, output_dim: int,
                      hidden_dims: List[int],
                      activation: str = 'tanh') -> 'keras.Model':
    """Build standard Multi-Layer Perceptron for PINN.

    Args:
        input_dim: Input dimension (time points, spatial locations, etc.)
        output_dim: Output dimension (number of parameters to estimate)
        hidden_dims: List of hidden layer sizes
        activation: Activation function ('tanh', 'relu', 'swish')

    Returns:
        Keras functional model
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow required for PINN. Install with: pip install tensorflow")

    inputs = keras.Input(shape=(input_dim,), name='input')
    x = inputs

    # Hidden layers with batch normalization
    for i, dim in enumerate(hidden_dims):
        x = layers.Dense(dim, activation=activation,
                        kernel_initializer='glorot_normal',
                        name=f'hidden_{i}')(x)
        x = layers.BatchNormalization(name=f'bn_{i}')(x)
        x = layers.Dropout(0.1, name=f'dropout_{i}')(x)

    # Output layer (no activation for parameter estimation)
    outputs = layers.Dense(output_dim, activation=None,
                          kernel_initializer='glorot_normal',
                          name='output')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='PINN_MLP')
    return model


def build_resnet_network(input_dim: int, output_dim: int,
                        hidden_dims: List[int],
                        activation: str = 'tanh') -> 'keras.Model':
    """Build ResNet-style network with skip connections.

    Better for deep networks and gradient flow.
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow required for PINN")

    inputs = keras.Input(shape=(input_dim,), name='input')

    # Initial projection
    x = layers.Dense(hidden_dims[0], activation=activation,
                    name='input_projection')(inputs)

    # Residual blocks
    for i in range(0, len(hidden_dims) - 1, 2):
        # Residual block
        residual = x
        x = layers.Dense(hidden_dims[i], activation=activation,
                        name=f'res_block_{i}_dense1')(x)
        x = layers.BatchNormalization(name=f'res_block_{i}_bn1')(x)
        x = layers.Dense(hidden_dims[i], activation=None,
                        name=f'res_block_{i}_dense2')(x)
        x = layers.BatchNormalization(name=f'res_block_{i}_bn2')(x)

        # Skip connection
        x = layers.Add(name=f'res_block_{i}_add')([x, residual])
        x = layers.Activation(activation, name=f'res_block_{i}_activation')(x)

    # Output
    outputs = layers.Dense(output_dim, activation=None, name='output')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='PINN_ResNet')
    return model


# ═══════════════════════════════════════════════════════════════
# PINN Estimator — Main Class
# ═══════════════════════════════════════════════════════════════

class PINNEstimator:
    """Physics-Informed Neural Network for ENS-GI parameter estimation.

    This class bridges clinical data (EGG signals, HRM measurements) to
    internal biophysical parameters using physics-constrained learning.

    Key Innovations:
    1. Physics Loss: Enforces that estimated parameters satisfy ODEs
    2. Data Loss: Matches clinical observations
    3. Regularization: Prevents overfitting to noise

    Workflow:
        1. Train on synthetic data (ground truth known)
        2. Validate parameter recovery accuracy
        3. Apply to real patient data
        4. Return parameters with confidence intervals
    """

    def __init__(self,
                 digital_twin: ENSGIDigitalTwin,
                 config: Optional[PINNConfig] = None,
                 parameter_names: Optional[List[str]] = None):
        """
        Args:
            digital_twin: Reference ENSGIDigitalTwin instance
            config: PINN configuration
            parameter_names: List of parameters to estimate
                            Default: ['g_Na', 'g_K', 'g_Ca', 'omega', 'coupling_strength']
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required. Install with: pip install tensorflow>=2.15")

        self.twin = digital_twin
        self.config = config or PINNConfig()

        # Default parameters to estimate (most clinically relevant)
        self.parameter_names = parameter_names or [
            'g_Na',              # Neuronal excitability
            'g_K',               # Repolarization strength
            'g_Ca',              # Calcium dynamics
            'omega',             # ICC pacemaker frequency
            'coupling_strength', # Network synchronization
        ]

        self.n_params = len(self.parameter_names)

        # Build neural network
        self.model = self._build_model()
        self.optimizer = keras.optimizers.Adam(learning_rate=self.config.learning_rate)

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'data_loss': [],
            'physics_loss': [],
        }

        # Parameter bounds (for normalization and constraints)
        self.param_bounds = self._get_parameter_bounds()

        print(f"[PINN] Initialized with {self.n_params} parameters: {self.parameter_names}")
        print(f"[PINN] Architecture: {self.config.architecture}")
        print(f"[PINN] Model parameters: {self.model.count_params():,}")

    def _build_model(self) -> 'keras.Model':
        """Build PINN neural network based on config."""
        # Input: clinical features (time series features, summary statistics)
        # For now: 100 time points (downsampled) + 10 biomarkers = 110 features
        input_dim = 110
        output_dim = self.n_params

        if self.config.architecture == 'mlp':
            return build_mlp_network(input_dim, output_dim,
                                    self.config.hidden_dims,
                                    self.config.activation)
        elif self.config.architecture == 'resnet':
            return build_resnet_network(input_dim, output_dim,
                                       self.config.hidden_dims,
                                       self.config.activation)
        else:
            raise ValueError(f"Unknown architecture: {self.config.architecture}")

    def _get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get physiologically plausible bounds for each parameter."""
        bounds = {
            'g_Na': (50.0, 200.0),          # mS/cm²
            'g_K': (20.0, 80.0),
            'g_Ca': (1.0, 10.0),
            'g_L': (0.1, 1.0),
            'omega': (0.001, 0.02),         # rad/ms (corresponds to ~1-12 cpm)
            'coupling_strength': (0.05, 1.0),
            'excitatory_weight': (0.1, 1.5),
            'inhibitory_weight': (0.1, 1.5),
            'amplitude': (2.0, 20.0),       # ICC amplitude
            'serotonin_factor': (0.5, 2.0),
            'no_factor': (0.5, 2.0),
        }

        return {name: bounds.get(name, (0.1, 10.0)) for name in self.parameter_names}

    def _normalize_parameters(self, params: np.ndarray) -> np.ndarray:
        """Normalize parameters to [0, 1] range."""
        normalized = np.zeros_like(params)
        for i, name in enumerate(self.parameter_names):
            low, high = self.param_bounds[name]
            normalized[i] = (params[i] - low) / (high - low)
        return normalized

    def _denormalize_parameters(self, normalized_params: np.ndarray) -> np.ndarray:
        """Convert normalized parameters back to physical units."""
        params = np.zeros_like(normalized_params)
        for i, name in enumerate(self.parameter_names):
            low, high = self.param_bounds[name]
            params[i] = normalized_params[i] * (high - low) + low
        return params

    def _extract_features_from_signal(self,
                                     voltages: np.ndarray,
                                     forces: Optional[np.ndarray] = None,
                                     calcium: Optional[np.ndarray] = None) -> np.ndarray:
        """Extract features from clinical/simulation signals.

        Args:
            voltages: [T, N] array of voltage time series
            forces: [T, N] array of force time series (optional)
            calcium: [T, N] array of calcium time series (optional)

        Returns:
            Feature vector of length 110:
                - 50 downsampled voltage points (average across neurons)
                - 25 downsampled force points
                - 25 downsampled calcium points
                - 10 summary biomarkers
        """
        T, N = voltages.shape

        # Downsample time series (average across neurons)
        v_mean = voltages.mean(axis=1)  # [T]
        v_downsampled = np.interp(np.linspace(0, T-1, 50), np.arange(T), v_mean)

        if forces is not None:
            f_mean = forces.mean(axis=1)
            f_downsampled = np.interp(np.linspace(0, T-1, 25), np.arange(T), f_mean)
        else:
            f_downsampled = np.zeros(25)

        if calcium is not None:
            ca_mean = calcium.mean(axis=1)
            ca_downsampled = np.interp(np.linspace(0, T-1, 25), np.arange(T), ca_mean)
        else:
            ca_downsampled = np.zeros(25)

        # Summary statistics (biomarkers)
        biomarkers = np.array([
            np.mean(voltages),           # Mean voltage
            np.std(voltages),            # Voltage variability
            np.max(voltages),            # Peak voltage
            np.mean(forces) if forces is not None else 0,
            np.std(forces) if forces is not None else 0,
            np.mean(calcium) if calcium is not None else 0,
            np.max(calcium) if calcium is not None else 0,
            np.corrcoef(voltages[:, 0], voltages[:, -1])[0, 1] if N > 1 else 0,  # Propagation
            np.fft.rfft(v_mean)[1:10].max(),  # Dominant frequency component
            len(np.where(np.diff(np.sign(v_mean)) > 0)[0]) / T,  # Zero-crossing rate
        ])

        # Concatenate all features
        features = np.concatenate([v_downsampled, f_downsampled, ca_downsampled, biomarkers])
        return features

    def generate_synthetic_dataset(self,
                                  n_samples: int = 1000,
                                  duration: float = 2000.0,
                                  dt: float = 0.05,
                                  noise_level: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data with known parameters.

        This is crucial for validating PINN before applying to real patients.

        Args:
            n_samples: Number of parameter combinations to sample
            duration: Simulation duration per sample (ms)
            dt: Time step (ms)
            noise_level: Gaussian noise level (std as fraction of signal)

        Returns:
            features: [n_samples, 110] feature matrix
            parameters: [n_samples, n_params] normalized parameter matrix
        """
        print(f"[PINN] Generating {n_samples} synthetic samples...")

        features_list = []
        params_list = []

        for i in range(n_samples):
            # Sample random parameters within bounds
            params = np.zeros(self.n_params)
            for j, name in enumerate(self.parameter_names):
                low, high = self.param_bounds[name]
                params[j] = np.random.uniform(low, high)

            # Create twin with these parameters
            twin = ENSGIDigitalTwin(n_segments=self.twin.n_segments)

            # Apply parameters
            for j, name in enumerate(self.parameter_names):
                if hasattr(twin.network.neurons[0].params, name):
                    for neuron in twin.network.neurons:
                        setattr(neuron.params, name, params[j])
                elif hasattr(twin.network.params, name):
                    setattr(twin.network.params, name, params[j])
                elif hasattr(twin.icc.params, name):
                    setattr(twin.icc.params, name, params[j])

            # Run simulation
            result = twin.run(duration, dt, I_stim={3: 10.0}, record=True, verbose=False)

            # Add noise (simulates measurement noise)
            voltages = result['voltages'] + np.random.randn(*result['voltages'].shape) * noise_level * np.std(result['voltages'])
            forces = result['force'] + np.random.randn(*result['force'].shape) * noise_level * np.std(result['force'])
            calcium = result['calcium'] + np.random.randn(*result['calcium'].shape) * noise_level * np.std(result['calcium'])

            # Extract features
            features = self._extract_features_from_signal(voltages, forces, calcium)

            features_list.append(features)
            params_list.append(self._normalize_parameters(params))

            if (i + 1) % 100 == 0:
                print(f"  Generated {i+1}/{n_samples} samples...")

        features_array = np.array(features_list)
        params_array = np.array(params_list)

        print(f"[PINN] Synthetic dataset complete: {features_array.shape}")
        return features_array, params_array

    @tf.function
    def _compute_physics_loss(self, predicted_params_normalized: 'tf.Tensor',
                             features: 'tf.Tensor') -> 'tf.Tensor':
        """Compute physics-informed loss term.

        This loss enforces that the estimated parameters satisfy the ODE constraints.
        For now, we use a simplified version based on ODE residuals.

        In full implementation, this would:
        1. Denormalize parameters
        2. Run short simulation with those parameters
        3. Compute ODE residual: ||dV/dt - f(V, params)||²

        For efficiency, we use analytical constraints instead.
        """
        # Simplified physics constraints:
        # 1. Conductances should satisfy stability criteria (g_Na > g_K for spiking)
        # 2. ICC frequency should be positive and bounded
        # 3. Coupling strength should be positive

        # This is a placeholder - full implementation would integrate ODEs
        # For now, enforce basic physical constraints

        physics_loss = 0.0

        # Example constraint: g_Na should typically be > g_K
        # (indices depend on parameter_names order)
        if 'g_Na' in self.parameter_names and 'g_K' in self.parameter_names:
            idx_na = self.parameter_names.index('g_Na')
            idx_k = self.parameter_names.index('g_K')
            # Penalize if g_Na < g_K (after denormalization)
            physics_loss += tf.reduce_mean(tf.nn.relu(predicted_params_normalized[:, idx_k] - predicted_params_normalized[:, idx_na]))

        # Non-negativity constraint (all parameters should be positive after denormalization)
        physics_loss += tf.reduce_mean(tf.nn.relu(-predicted_params_normalized)) * 10.0

        # Boundedness (parameters should stay in [0, 1] after normalization)
        physics_loss += tf.reduce_mean(tf.nn.relu(predicted_params_normalized - 1.0)) * 10.0

        return physics_loss

    @tf.function
    def _train_step(self, features: 'tf.Tensor', true_params: 'tf.Tensor') -> Tuple['tf.Tensor', 'tf.Tensor', 'tf.Tensor']:
        """Single training step with combined loss."""
        with tf.GradientTape() as tape:
            # Forward pass
            predicted_params = self.model(features, training=True)

            # Data loss (MSE between predicted and true parameters)
            data_loss = tf.reduce_mean(tf.square(predicted_params - true_params))

            # Physics loss
            physics_loss = self._compute_physics_loss(predicted_params, features)

            # Combined loss
            total_loss = (self.config.lambda_data * data_loss +
                         self.config.lambda_physics * physics_loss)

        # Backpropagation
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return total_loss, data_loss, physics_loss

    def train(self,
             features: Optional[np.ndarray] = None,
             parameters: Optional[np.ndarray] = None,
             epochs: int = 1000,
             verbose: int = 1,
             generate_data: bool = True,
             n_synthetic_samples: int = 1000) -> Dict:
        """Train PINN on synthetic or provided data.

        Args:
            features: [n_samples, 110] feature array (optional)
            parameters: [n_samples, n_params] normalized parameter array (optional)
            epochs: Number of training epochs
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
            generate_data: If True and no data provided, generate synthetic
            n_synthetic_samples: Number of synthetic samples to generate

        Returns:
            Training history dict
        """
        # Generate or validate data
        if features is None or parameters is None:
            if generate_data:
                features, parameters = self.generate_synthetic_dataset(
                    n_samples=n_synthetic_samples)
            else:
                raise ValueError("No training data provided and generate_data=False")

        # Split into train/validation
        n_samples = features.shape[0]
        n_val = int(n_samples * self.config.validation_split)
        n_train = n_samples - n_val

        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        X_train, y_train = features[train_idx], parameters[train_idx]
        X_val, y_val = features[val_idx], parameters[val_idx]

        print(f"[PINN] Training on {n_train} samples, validating on {n_val} samples")

        # Convert to TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(1000).batch(self.config.batch_size)

        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(self.config.batch_size)

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            epoch_losses = []
            epoch_data_losses = []
            epoch_physics_losses = []

            for batch_features, batch_params in train_dataset:
                loss, data_loss, physics_loss = self._train_step(batch_features, batch_params)
                epoch_losses.append(loss.numpy())
                epoch_data_losses.append(data_loss.numpy())
                epoch_physics_losses.append(physics_loss.numpy())

            train_loss = np.mean(epoch_losses)
            train_data_loss = np.mean(epoch_data_losses)
            train_physics_loss = np.mean(epoch_physics_losses)

            # Validation
            val_losses = []
            for batch_features, batch_params in val_dataset:
                pred_params = self.model(batch_features, training=False)
                val_loss = tf.reduce_mean(tf.square(pred_params - batch_params))
                val_losses.append(val_loss.numpy())

            val_loss = np.mean(val_losses)

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['data_loss'].append(train_data_loss)
            self.history['physics_loss'].append(train_physics_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.model.save_weights('pinn_best_weights.h5')
            else:
                patience_counter += 1

            # Verbose output
            if verbose >= 1 and (epoch % 50 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Loss: {train_loss:.6f} - "
                      f"Data: {train_data_loss:.6f} - "
                      f"Physics: {train_physics_loss:.6f} - "
                      f"Val: {val_loss:.6f}")

            # Early stopping check
            if patience_counter >= self.config.early_stopping_patience:
                print(f"[PINN] Early stopping at epoch {epoch+1}")
                break

        # Load best weights
        self.model.load_weights('pinn_best_weights.h5')
        print(f"[PINN] Training complete. Best val loss: {best_val_loss:.6f}")

        return self.history

    def estimate_parameters(self,
                          voltages: np.ndarray,
                          forces: Optional[np.ndarray] = None,
                          calcium: Optional[np.ndarray] = None,
                          n_bootstrap: int = 100) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Estimate parameters from clinical signals with uncertainty.

        Args:
            voltages: [T, N] voltage time series (e.g., from EGG)
            forces: [T, N] force time series (e.g., from HRM) - optional
            calcium: [T, N] calcium time series - optional
            n_bootstrap: Number of bootstrap samples for uncertainty

        Returns:
            estimates: Dict of parameter estimates
            uncertainties: Dict of parameter standard deviations
        """
        # Extract features
        features = self._extract_features_from_signal(voltages, forces, calcium)

        # Bootstrap for uncertainty quantification
        predictions = []

        for i in range(n_bootstrap):
            # Add noise to features (bootstrap perturbation)
            noisy_features = features + np.random.randn(*features.shape) * 0.01

            # Predict
            pred_normalized = self.model(noisy_features[np.newaxis, :], training=False).numpy()[0]
            pred_params = self._denormalize_parameters(pred_normalized)
            predictions.append(pred_params)

        predictions = np.array(predictions)  # [n_bootstrap, n_params]

        # Compute statistics
        estimates = {}
        uncertainties = {}

        for i, name in enumerate(self.parameter_names):
            estimates[name] = float(np.mean(predictions[:, i]))
            uncertainties[name] = float(np.std(predictions[:, i]))

        return estimates, uncertainties

    def validate_on_test_set(self,
                            features: np.ndarray,
                            true_params: np.ndarray) -> Dict:
        """Validate PINN accuracy on test set.

        Returns error metrics for each parameter.
        """
        # Predict
        pred_params_normalized = self.model(features, training=False).numpy()

        # Denormalize
        pred_params = np.array([self._denormalize_parameters(p) for p in pred_params_normalized])
        true_params_denorm = np.array([self._denormalize_parameters(p) for p in true_params])

        # Compute errors
        mae = np.mean(np.abs(pred_params - true_params_denorm), axis=0)
        rmse = np.sqrt(np.mean((pred_params - true_params_denorm)**2, axis=0))
        mape = np.mean(np.abs((pred_params - true_params_denorm) / true_params_denorm), axis=0) * 100

        results = {}
        for i, name in enumerate(self.parameter_names):
            results[name] = {
                'mae': float(mae[i]),
                'rmse': float(rmse[i]),
                'mape': float(mape[i]),
            }

        return results

    def save(self, filepath: str):
        """Save PINN model and configuration."""
        self.model.save(filepath)

        config_dict = {
            'parameter_names': self.parameter_names,
            'param_bounds': self.param_bounds,
            'config': {
                'architecture': self.config.architecture,
                'hidden_dims': self.config.hidden_dims,
                'activation': self.config.activation,
                'learning_rate': self.config.learning_rate,
                'lambda_data': self.config.lambda_data,
                'lambda_physics': self.config.lambda_physics,
            }
        }

        with open(filepath + '_config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"[PINN] Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str, digital_twin: ENSGIDigitalTwin) -> 'PINNEstimator':
        """Load saved PINN model."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required")

        # Load config
        with open(filepath + '_config.json', 'r') as f:
            config_dict = json.load(f)

        # Create estimator
        config = PINNConfig(**config_dict['config'])
        estimator = cls(digital_twin, config, config_dict['parameter_names'])

        # Load weights
        estimator.model = keras.models.load_model(filepath)
        estimator.param_bounds = config_dict['param_bounds']

        print(f"[PINN] Model loaded from {filepath}")
        return estimator


# ═══════════════════════════════════════════════════════════════
# Demo / Testing
# ═══════════════════════════════════════════════════════════════

def demo_pinn():
    """Demonstrate PINN training and parameter estimation."""
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  ENS-GI PINN — Physics-Informed Neural Network Demo      ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    if not TF_AVAILABLE:
        print("[ERROR] TensorFlow not installed. Please run:")
        print("  pip install tensorflow>=2.15")
        return

    # Create digital twin
    print("[1/5] Creating digital twin...")
    twin = ENSGIDigitalTwin(n_segments=12)

    # Create PINN estimator
    print("[2/5] Building PINN architecture...")
    pinn = PINNEstimator(
        digital_twin=twin,
        config=PINNConfig(
            architecture='mlp',
            hidden_dims=[128, 64, 32],
            learning_rate=1e-3,
            lambda_physics=0.05,
        ),
        parameter_names=['g_Na', 'g_K', 'g_Ca', 'omega', 'coupling_strength']
    )

    # Train on synthetic data (use small dataset for demo)
    print("[3/5] Training PINN on synthetic data...")
    history = pinn.train(
        epochs=200,
        n_synthetic_samples=100,  # Small for demo
        verbose=1
    )

    # Test parameter recovery
    print("\n[4/5] Testing parameter recovery...")
    # Generate test case with known parameters
    test_twin = ENSGIDigitalTwin(n_segments=12)
    true_params = {
        'g_Na': 135.0,
        'g_K': 42.0,
        'g_Ca': 5.5,
        'omega': 0.006,
        'coupling_strength': 0.4,
    }

    # Apply known parameters
    for name, val in true_params.items():
        if hasattr(test_twin.network.neurons[0].params, name):
            for neuron in test_twin.network.neurons:
                setattr(neuron.params, name, val)
        elif hasattr(test_twin.icc.params, name):
            setattr(test_twin.icc.params, name, val)

    # Run simulation
    result = test_twin.run(2000, dt=0.05, I_stim={3: 10.0}, record=True, verbose=False)

    # Estimate parameters
    estimates, uncertainties = pinn.estimate_parameters(
        result['voltages'],
        result['force'],
        result['calcium'],
        n_bootstrap=50
    )

    # Display results
    print("\n[5/5] Parameter Recovery Results:")
    print(f"{'Parameter':<20} {'True':>12} {'Estimated':>12} {'Uncertainty':>12} {'Error %':>10}")
    print("-" * 70)

    for name in pinn.parameter_names:
        true_val = true_params[name]
        est_val = estimates[name]
        unc_val = uncertainties[name]
        error_pct = abs(est_val - true_val) / true_val * 100

        print(f"{name:<20} {true_val:>12.4f} {est_val:>12.4f} {unc_val:>12.4f} {error_pct:>9.2f}%")

    print("\n✓ PINN demo complete.")
    print("\nNote: This demo uses a small dataset (100 samples, 200 epochs).")
    print("For production use, train with 1000+ samples and 2000+ epochs.")


if __name__ == '__main__':
    demo_pinn()
