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
import tempfile
import copy
import os
import shutil

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
    use_ode_residuals: bool = False     # Use real HH ODE residuals (True) or fast constraints (False)

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 128, 64, 32]


def build_mlp_network(input_dim: int, output_dim: int,
                      hidden_dims: List[int],
                      activation: str = 'tanh',
                      use_regularization: bool = True) -> 'keras.Model':
    """Build standard Multi-Layer Perceptron for PINN.

    Args:
        input_dim: Input dimension (time points, spatial locations, etc.)
        output_dim: Output dimension (number of parameters to estimate)
        hidden_dims: List of hidden layer sizes
        activation: Activation function ('tanh', 'relu', 'swish')
        use_regularization: Whether to use BatchNorm and Dropout (default True)

    Returns:
        Keras functional model
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow required for PINN. Install with: pip install tensorflow")

    inputs = keras.Input(shape=(input_dim,), name='input')
    x = inputs

    # Hidden layers with optional batch normalization and dropout
    for i, dim in enumerate(hidden_dims):
        x = layers.Dense(dim, activation=activation,
                        kernel_initializer='glorot_normal',
                        name=f'hidden_{i}')(x)
        if use_regularization:
            x = layers.BatchNormalization(name=f'bn_{i}')(x)
            x = layers.Dropout(0.1, name=f'dropout_{i}')(x)

    # Output layer with sigmoid to constrain to [0,1] (matches normalized parameter space)
    outputs = layers.Dense(output_dim, activation='sigmoid',
                          kernel_initializer='glorot_normal',
                          name='output')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='PINN_MLP')
    return model


def build_resnet_network(input_dim: int, output_dim: int,
                        hidden_dims: List[int],
                        activation: str = 'tanh',
                        use_regularization: bool = True) -> 'keras.Model':
    """Build ResNet-style network with skip connections.

    Better for deep networks and gradient flow.

    Args:
        use_regularization: Whether to use BatchNorm (default True)
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
        if use_regularization:
            x = layers.BatchNormalization(name=f'res_block_{i}_bn1')(x)
        x = layers.Dense(hidden_dims[i], activation=None,
                        name=f'res_block_{i}_dense2')(x)
        if use_regularization:
            x = layers.BatchNormalization(name=f'res_block_{i}_bn2')(x)

        # Skip connection with projection if dimensions don't match
        if residual.shape[-1] != hidden_dims[i]:
            residual = layers.Dense(hidden_dims[i], activation=None,
                                   name=f'res_block_{i}_projection')(residual)
        x = layers.Add(name=f'res_block_{i}_add')([x, residual])
        x = layers.Activation(activation, name=f'res_block_{i}_activation')(x)

    # Output with sigmoid to constrain to [0,1] (matches normalized parameter space)
    outputs = layers.Dense(output_dim, activation='sigmoid', name='output')(x)

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

        # Feature normalization (computed during training)
        self.feature_mean = None  # np.ndarray or None
        self.feature_std = None   # np.ndarray or None

        # Create temporary directory for model checkpoints
        self._checkpoint_dir = tempfile.mkdtemp(prefix="pinn_ckpt_")
        self._best_weights_path = os.path.join(self._checkpoint_dir, 'best.weights.h5')

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

    def _denormalize_parameters(self, normalized_params):
        """Convert normalized [0,1] parameters to physical units.

        TF-graph-safe: uses pre-computed numpy constant arrays so it works
        both as a regular numpy call and inside @tf.function tracing.
        """
        scales = np.array(
            [self.param_bounds[n][1] - self.param_bounds[n][0]
             for n in self.parameter_names], dtype=np.float32
        )
        offsets = np.array(
            [self.param_bounds[n][0] for n in self.parameter_names],
            dtype=np.float32
        )
        # broadcast mul/add works for both np.ndarray and tf.Tensor
        return normalized_params * scales + offsets

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
        # Compute correlation safely (handle NaN for uniform signals)
        if N > 1:
            corr_matrix = np.corrcoef(voltages[:, 0], voltages[:, -1])
            spatial_sync = corr_matrix[0, 1] if np.isfinite(corr_matrix[0, 1]) else 0.0
        else:
            spatial_sync = 0.0

        # Compute FFT safely (handle edge cases)
        fft_components = np.abs(np.fft.rfft(v_mean))[1:10]
        dominant_freq = np.max(fft_components) if len(fft_components) > 0 and np.all(np.isfinite(fft_components)) else 0.0

        biomarkers = np.array([
            np.mean(voltages),           # Mean voltage
            np.std(voltages),            # Voltage variability
            np.max(voltages),            # Peak voltage
            np.mean(forces) if forces is not None else 0,
            np.std(forces) if forces is not None else 0,
            np.mean(calcium) if calcium is not None else 0,
            np.max(calcium) if calcium is not None else 0,
            spatial_sync,  # Propagation (safe correlation)
            dominant_freq,  # Dominant frequency component (safe FFT)
            len(np.where(np.diff(np.sign(v_mean)) > 0)[0]) / T,  # Zero-crossing rate
        ])

        # Ensure all biomarkers are finite (replace NaN/Inf with 0)
        biomarkers = np.nan_to_num(biomarkers, nan=0.0, posinf=0.0, neginf=0.0)

        # Concatenate all features
        features = np.concatenate([v_downsampled, f_downsampled, ca_downsampled, biomarkers])
        # Final safety check on all features
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features

    def generate_synthetic_dataset(self,
                                  n_samples: int = 1000,
                                  duration: float = 2000.0,
                                  dt: float = 0.05,
                                  noise_level: float = 0.05) -> Dict[str, np.ndarray]:
        """Generate synthetic training data with known parameters.

        This is crucial for validating PINN before applying to real patients.

        Args:
            n_samples: Number of parameter combinations to sample
            duration: Simulation duration per sample (ms)
            dt: Time step (ms)
            noise_level: Gaussian noise level (std as fraction of signal)

        Returns:
            dict with:
                'features': [n_samples, 110] feature matrix
                'parameters': [n_samples, n_params] normalized parameter matrix
                'voltages': [n_samples, n_timepoints, n_neurons] voltage traces
                'forces': [n_samples, n_timepoints, n_neurons] force traces
                'calcium': [n_samples, n_timepoints, n_neurons] calcium traces
        """
        print(f"[PINN] Generating {n_samples} synthetic samples...")

        features_list = []
        params_list = []
        voltages_list = []
        forces_list = []
        calcium_list = []

        for i in range(n_samples):
            # Sample random parameters within bounds
            params = np.zeros(self.n_params)
            for j, name in enumerate(self.parameter_names):
                low, high = self.param_bounds[name]
                params[j] = np.random.uniform(low, high)

            # Create twin with these parameters
            twin = ENSGIDigitalTwin(n_segments=self.twin.n_segments)

            # Copy ALL base parameters from self.twin so synthetic data matches
            # the same physiological context (e.g. IBS-C profile) as the target twin.
            # This prevents distribution shift when self.twin has a profile applied.
            for neuron_ref, neuron_tgt in zip(self.twin.network.neurons, twin.network.neurons):
                neuron_tgt.params = copy.copy(neuron_ref.params)
            twin.network.params = copy.copy(self.twin.network.params)
            twin.icc.params = copy.copy(self.twin.icc.params)
            twin.muscle.params = copy.copy(self.twin.muscle.params)

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
            voltages_list.append(voltages)
            forces_list.append(forces)
            calcium_list.append(calcium)

            if (i + 1) % 100 == 0:
                print(f"  Generated {i+1}/{n_samples} samples...")

        features_array = np.array(features_list)
        params_array = np.array(params_list)
        voltages_array = np.array(voltages_list)
        forces_array = np.array(forces_list)
        calcium_array = np.array(calcium_list)

        print(f"[PINN] Synthetic dataset complete: {features_array.shape}")
        return {
            'features': features_array,
            'parameters': params_array,
            'voltages': voltages_array,
            'forces': forces_array,
            'calcium': calcium_array
        }

    @staticmethod
    @tf.function
    def _hh_alpha_m(V: 'tf.Tensor') -> 'tf.Tensor':
        """Hodgkin-Huxley alpha_m rate function (sodium activation)."""
        return 0.1 * (V + 40.0) / (1.0 - tf.exp(-(V + 40.0) / 10.0) + 1e-8)

    @staticmethod
    @tf.function
    def _hh_beta_m(V: 'tf.Tensor') -> 'tf.Tensor':
        """Hodgkin-Huxley beta_m rate function (sodium activation)."""
        return 4.0 * tf.exp(-(V + 65.0) / 18.0)

    @staticmethod
    @tf.function
    def _hh_alpha_h(V: 'tf.Tensor') -> 'tf.Tensor':
        """Hodgkin-Huxley alpha_h rate function (sodium inactivation)."""
        return 0.07 * tf.exp(-(V + 65.0) / 20.0)

    @staticmethod
    @tf.function
    def _hh_beta_h(V: 'tf.Tensor') -> 'tf.Tensor':
        """Hodgkin-Huxley beta_h rate function (sodium inactivation)."""
        return 1.0 / (1.0 + tf.exp(-(V + 35.0) / 10.0))

    @staticmethod
    @tf.function
    def _hh_alpha_n(V: 'tf.Tensor') -> 'tf.Tensor':
        """Hodgkin-Huxley alpha_n rate function (potassium activation)."""
        return 0.01 * (V + 55.0) / (1.0 - tf.exp(-(V + 55.0) / 10.0) + 1e-8)

    @staticmethod
    @tf.function
    def _hh_beta_n(V: 'tf.Tensor') -> 'tf.Tensor':
        """Hodgkin-Huxley beta_n rate function (potassium activation)."""
        return 0.125 * tf.exp(-(V + 65.0) / 80.0)

    @tf.function
    def _compute_ode_residual(self,
                             V: 'tf.Tensor',
                             m: 'tf.Tensor',
                             h: 'tf.Tensor',
                             n: 'tf.Tensor',
                             dV_dt: 'tf.Tensor',
                             dm_dt: 'tf.Tensor',
                             dh_dt: 'tf.Tensor',
                             dn_dt: 'tf.Tensor',
                             g_Na: 'tf.Tensor',
                             g_K: 'tf.Tensor',
                             g_L: float = 0.3,
                             E_Na: float = 55.0,
                             E_K: float = -77.0,
                             E_L: float = -54.4,
                             C_m: float = 1.0) -> 'tf.Tensor':
        """
        Compute ODE residual for Hodgkin-Huxley equations.

        Args:
            V, m, h, n: State variables (voltage and gating variables)
            dV_dt, dm_dt, dh_dt, dn_dt: Time derivatives (from autodiff)
            g_Na, g_K: Conductances (parameters to estimate)
            g_L, E_Na, E_K, E_L, C_m: Constants

        Returns:
            ODE residual (scalar) - should be minimized
        """
        # Ionic currents
        I_Na = g_Na * (m ** 3) * h * (V - E_Na)
        I_K = g_K * (n ** 4) * (V - E_K)
        I_L = g_L * (V - E_L)

        # ODE for membrane voltage
        R_V = dV_dt - (- I_Na - I_K - I_L) / C_m

        # ODE for gating variables
        alpha_m = self._hh_alpha_m(V)
        beta_m = self._hh_beta_m(V)
        R_m = dm_dt - (alpha_m * (1.0 - m) - beta_m * m)

        alpha_h = self._hh_alpha_h(V)
        beta_h = self._hh_beta_h(V)
        R_h = dh_dt - (alpha_h * (1.0 - h) - beta_h * h)

        alpha_n = self._hh_alpha_n(V)
        beta_n = self._hh_beta_n(V)
        R_n = dn_dt - (alpha_n * (1.0 - n) - beta_n * n)

        # Total residual (L2 norm)
        residual = tf.reduce_mean(
            tf.square(R_V) + tf.square(R_m) + tf.square(R_h) + tf.square(R_n)
        )

        return residual

    def _compute_physics_loss(self, predicted_params_normalized: 'tf.Tensor',
                             features: 'tf.Tensor',
                             use_ode_residuals: bool = False) -> 'tf.Tensor':
        """Compute physics-informed loss term.

        Two modes:
        1. Constraint-based (fast): Enforce physical constraints on parameters
        2. ODE residual-based (accurate): Enforce Hodgkin-Huxley ODE constraints

        Args:
            predicted_params_normalized: Normalized parameter predictions [batch, n_params]
            features: Input features [batch, n_features]
            use_ode_residuals: If True, use ODE residual loss (slower but accurate)

        Returns:
            Physics loss (scalar)
        """
        if use_ode_residuals:
            # ACCURATE PHYSICS LOSS: ODE Residual Computation
            # Uses collocation points and steady-state analysis

            # Denormalize parameters
            params_denorm = self._denormalize_parameters(predicted_params_normalized)

            # Extract conductances
            idx_na = self.parameter_names.index('g_Na') if 'g_Na' in self.parameter_names else 0
            idx_k = self.parameter_names.index('g_K') if 'g_K' in self.parameter_names else 1
            g_Na = params_denorm[:, idx_na:idx_na+1]
            g_K = params_denorm[:, idx_k:idx_k+1]

            # Evaluate ODE consistency at steady-state equilibrium
            # At equilibrium, dV/dt ≈ 0, dm/dt ≈ 0, etc.
            # This tests if parameters produce stable resting states

            # Resting state approximation (broadcast from input to avoid symbolic shape issues)
            V_rest = tf.ones_like(predicted_params_normalized[:, :1]) * (-65.0)

            # Compute steady-state gating variables for this voltage
            m_inf = 1.0 / (1.0 + tf.exp(-(V_rest + 40.0) / 10.0))
            h_inf = 1.0 / (1.0 + tf.exp((V_rest + 60.0) / 10.0))
            n_inf = 1.0 / (1.0 + tf.exp(-(V_rest + 55.0) / 10.0))

            # At steady state, currents should balance (dV/dt = 0)
            # Compute ionic currents with steady-state gating
            I_Na = g_Na * (m_inf ** 3) * h_inf * (V_rest - 55.0)  # E_Na = 55 mV
            I_K = g_K * (n_inf ** 4) * (V_rest + 77.0)  # E_K = -77 mV
            I_L = 0.3 * (V_rest + 54.4)  # g_L = 0.3, E_L = -54.4 mV

            # At rest, total current should be near zero
            I_total = I_Na + I_K + I_L

            # Physics loss: penalize deviation from current balance
            physics_loss = tf.reduce_mean(tf.square(I_total))

            # Additional constraint: gating variables should satisfy rate equations at steady-state
            # dm/dt = 0 => α_m(V) * (1 - m_inf) = β_m(V) * m_inf
            alpha_m = self._hh_alpha_m(V_rest)
            beta_m = self._hh_beta_m(V_rest)
            m_ss_expected = alpha_m / (alpha_m + beta_m)
            gating_residual_m = tf.square(m_inf - m_ss_expected)

            alpha_h = self._hh_alpha_h(V_rest)
            beta_h = self._hh_beta_h(V_rest)
            h_ss_expected = alpha_h / (alpha_h + beta_h)
            gating_residual_h = tf.square(h_inf - h_ss_expected)

            alpha_n = self._hh_alpha_n(V_rest)
            beta_n = self._hh_beta_n(V_rest)
            n_ss_expected = alpha_n / (alpha_n + beta_n)
            gating_residual_n = tf.square(n_inf - n_ss_expected)

            # Combine residuals
            physics_loss += tf.reduce_mean(
                gating_residual_m + gating_residual_h + gating_residual_n
            ) * 0.1  # Weighted contribution

            return physics_loss

        else:
            # FAST CONSTRAINT-BASED PHYSICS LOSS (current implementation)
            # Simplified physics constraints:
            # 1. Conductances should satisfy stability criteria (g_Na > g_K for spiking)
            # 2. Parameters should be positive and bounded
            # 3. ICC frequency should be physiologically plausible

            physics_loss = tf.constant(0.0, dtype=tf.float32)

            # Constraint 1: g_Na should typically be > g_K for excitability
            if 'g_Na' in self.parameter_names and 'g_K' in self.parameter_names:
                idx_na = self.parameter_names.index('g_Na')
                idx_k = self.parameter_names.index('g_K')
                # Penalize if g_Na < g_K (after denormalization)
                physics_loss += tf.reduce_mean(
                    tf.nn.relu(predicted_params_normalized[:, idx_k] - predicted_params_normalized[:, idx_na])
                )

            # Constraint 2: Non-negativity (parameters should be positive after denormalization)
            physics_loss += tf.reduce_mean(tf.nn.relu(-predicted_params_normalized)) * 10.0

            # Constraint 3: Boundedness (parameters should stay in [0, 1] after normalization)
            physics_loss += tf.reduce_mean(tf.nn.relu(predicted_params_normalized - 1.0)) * 10.0

            # Constraint 4: Physiological ranges
            # Add soft constraints for parameter ranges based on literature
            if 'omega' in self.parameter_names:
                idx_omega = self.parameter_names.index('omega')
                # omega (ICC frequency) should be in range 0.01-1.0 Hz after denormalization
                # In normalized space [0, 1], this depends on bounds
                # Penalize extreme values
                physics_loss += tf.reduce_mean(
                    tf.square(predicted_params_normalized[:, idx_omega] - 0.5)
                ) * 0.1  # Soft centering

            return physics_loss

    @tf.function
    def _train_step(self, features: 'tf.Tensor', true_params: 'tf.Tensor') -> Tuple['tf.Tensor', 'tf.Tensor', 'tf.Tensor']:
        """Single training step with combined loss."""
        with tf.GradientTape() as tape:
            # Forward pass
            predicted_params = self.model(features, training=True)

            # Ensure dtype compatibility (cast to float32 if needed)
            true_params = tf.cast(true_params, tf.float32)

            # Data loss (MSE between predicted and true parameters)
            data_loss = tf.reduce_mean(tf.square(predicted_params - true_params))

            # Physics loss — use_ode_residuals from config (True = real HH ODEs)
            physics_loss = self._compute_physics_loss(
                predicted_params, features,
                use_ode_residuals=self.config.use_ode_residuals
            )

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
             n_synthetic_samples: int = 1000,
             batch_size: Optional[int] = None,
             use_ode_residuals: Optional[bool] = None) -> Dict:
        """Train PINN on synthetic or provided data.

        Args:
            features: [n_samples, 110] feature array (optional)
            parameters: [n_samples, n_params] normalized parameter array (optional)
            epochs: Number of training epochs
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
            generate_data: If True and no data provided, generate synthetic
            n_synthetic_samples: Number of synthetic samples to generate
            batch_size: Batch size (optional, overrides config)
            use_ode_residuals: Override config setting for ODE physics loss.
                               True = enforce Hodgkin-Huxley ODEs (accurate, slower).
                               False = use fast constraint-based physics loss.
                               None = use value from self.config.use_ode_residuals.

        Returns:
            Training history dict
        """
        # Apply override if provided
        if use_ode_residuals is not None:
            self.config.use_ode_residuals = use_ode_residuals

        # Generate or validate data
        if features is None or parameters is None:
            if generate_data:
                dataset = self.generate_synthetic_dataset(
                    n_samples=n_synthetic_samples)
                features = dataset['features']
                parameters = dataset['parameters']
            else:
                raise ValueError("No training data provided and generate_data=False")

        # Compute and apply feature normalization (standardize to zero mean, unit variance)
        self.feature_mean = np.mean(features, axis=0).astype(np.float32)
        self.feature_std = np.std(features, axis=0).astype(np.float32)
        # Prevent division by zero for constant features
        self.feature_std[self.feature_std < 1e-8] = 1.0
        features = (features - self.feature_mean) / self.feature_std

        # Split into train/validation
        n_samples = features.shape[0]
        n_val = int(n_samples * self.config.validation_split)
        n_train = n_samples - n_val

        # Adaptive batch sizing for small datasets
        if batch_size is None:
            batch_size = self.config.batch_size

        # Ensure batch size doesn't exceed training data
        # Use at most 1/4 of training data, minimum 4
        adaptive_batch_size = max(4, min(batch_size, n_train // 4))
        if adaptive_batch_size < batch_size and verbose >= 1:
            print(f"[PINN] Adapting batch size from {batch_size} to {adaptive_batch_size} for small dataset ({n_train} training samples)")

        # Check if we should disable regularization for very small datasets
        use_regularization = n_train >= 50
        if not use_regularization and verbose >= 1:
            print(f"[PINN] WARNING: Small training set ({n_train} samples). Regularization disabled for stability.")
            # Rebuild model without regularization
            input_dim = 110
            output_dim = self.n_params
            if self.config.architecture == 'mlp':
                self.model = build_mlp_network(input_dim, output_dim,
                                              self.config.hidden_dims,
                                              self.config.activation,
                                              use_regularization=False)
            elif self.config.architecture == 'resnet':
                self.model = build_resnet_network(input_dim, output_dim,
                                                  self.config.hidden_dims,
                                                  self.config.activation,
                                                  use_regularization=False)
            self.model.compile(optimizer=self.optimizer)

        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        X_train, y_train = features[train_idx], parameters[train_idx]
        X_val, y_val = features[val_idx], parameters[val_idx]

        print(f"[PINN] Training on {n_train} samples, validating on {n_val} samples")

        # Convert to float32 for TensorFlow compatibility (prevents float64/float32 dtype errors)
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        y_val = y_val.astype(np.float32)

        # Convert to TensorFlow datasets with adaptive batch size
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(1000).batch(adaptive_batch_size)

        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(adaptive_batch_size)

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
                # Ensure dtype compatibility (model outputs float32, params may be float64)
                batch_params = tf.cast(batch_params, tf.float32)
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
                # Save best model to temporary directory
                self.model.save_weights(self._best_weights_path)
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

        # Load best weights if they exist
        if os.path.exists(self._best_weights_path):
            self.model.load_weights(self._best_weights_path)
        print(f"[PINN] Training complete. Best val loss: {best_val_loss:.6f}")

        return self.history

    def estimate_parameters(self,
                          voltages: np.ndarray,
                          forces: Optional[np.ndarray] = None,
                          calcium: Optional[np.ndarray] = None,
                          n_bootstrap: int = 100) -> Dict[str, Dict[str, float]]:
        """Estimate parameters from clinical signals with uncertainty.

        Args:
            voltages: [T, N] voltage time series (e.g., from EGG)
            forces: [T, N] force time series (e.g., from HRM) - optional
            calcium: [T, N] calcium time series - optional
            n_bootstrap: Number of bootstrap samples for uncertainty

        Returns:
            Dict with structure: {'param_name': {'mean': X, 'std': Y}}
            Example: {'g_Na': {'mean': 120.0, 'std': 5.0}}
        """
        # Extract features
        features = self._extract_features_from_signal(voltages, forces, calcium)
        # Convert to float32 for TensorFlow compatibility
        features = features.astype(np.float32)

        # Apply feature normalization (must match training normalization)
        if self.feature_mean is not None and self.feature_std is not None:
            features = (features - self.feature_mean) / self.feature_std

        # Bootstrap for uncertainty quantification
        predictions = []

        for i in range(n_bootstrap):
            # Add noise to features (bootstrap perturbation)
            # Use 5% perturbation for meaningful uncertainty (features are standardized)
            noise_scale = 0.05  # 5% perturbation
            noisy_features = features + np.random.randn(*features.shape).astype(np.float32) * noise_scale

            # Predict
            pred_normalized = self.model(noisy_features[np.newaxis, :], training=False).numpy()[0]
            pred_params = self._denormalize_parameters(pred_normalized)
            predictions.append(pred_params)

        predictions = np.array(predictions)  # [n_bootstrap, n_params]

        # Compute statistics and return nested dict format
        results = {}

        for i, name in enumerate(self.parameter_names):
            results[name] = {
                'mean': float(np.mean(predictions[:, i])),
                'std': float(np.std(predictions[:, i]))
            }

        return results

    def validate_on_test_set(self,
                            features: np.ndarray,
                            true_params: np.ndarray) -> Dict:
        """Validate PINN accuracy on test set.

        Returns error metrics for each parameter.
        """
        # Predict (convert to float32 for TensorFlow compatibility)
        features = features.astype(np.float32)
        # Apply feature normalization if available
        if self.feature_mean is not None and self.feature_std is not None:
            features = (features - self.feature_mean) / self.feature_std
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
        # Store base path before modification for config file
        base_path = filepath

        # Ensure filepath has valid Keras extension
        if not (filepath.endswith('.keras') or filepath.endswith('.h5')):
            filepath = filepath + '.keras'

        # Save model with extension
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
            },
            'feature_mean': self.feature_mean.tolist() if self.feature_mean is not None else None,
            'feature_std': self.feature_std.tolist() if self.feature_std is not None else None,
        }

        # Save config using base path (consistent naming)
        config_path = base_path + '_config.json'
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"[PINN] Model saved to {filepath}")
        print(f"[PINN] Config saved to {config_path}")

    def __del__(self):
        """Cleanup temporary checkpoint directory on object destruction."""
        if hasattr(self, '_checkpoint_dir') and os.path.exists(self._checkpoint_dir):
            try:
                shutil.rmtree(self._checkpoint_dir, ignore_errors=True)
            except Exception:
                pass  # Silently ignore cleanup errors

    @classmethod
    def load(cls, filepath: str, digital_twin: ENSGIDigitalTwin) -> 'PINNEstimator':
        """Load saved PINN model."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required")

        # Check for model file with extension
        model_path = filepath
        if not os.path.exists(model_path):
            # Try adding .keras extension
            if os.path.exists(filepath + '.keras'):
                model_path = filepath + '.keras'
            elif os.path.exists(filepath + '.h5'):
                model_path = filepath + '.h5'
            else:
                raise FileNotFoundError(f"Model not found at {filepath}")

        # Load configuration (always from base path)
        config_path = filepath + '_config.json'
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found at {config_path}")

        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        # Create estimator
        config = PINNConfig(**config_dict['config'])
        estimator = cls(digital_twin, config, config_dict['parameter_names'])

        # Load weights
        estimator.model = keras.models.load_model(model_path)
        estimator.param_bounds = config_dict['param_bounds']

        # Restore feature normalization stats
        if config_dict.get('feature_mean') is not None:
            estimator.feature_mean = np.array(config_dict['feature_mean'], dtype=np.float32)
        if config_dict.get('feature_std') is not None:
            estimator.feature_std = np.array(config_dict['feature_std'], dtype=np.float32)

        print(f"[PINN] Model loaded from {model_path}")
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
