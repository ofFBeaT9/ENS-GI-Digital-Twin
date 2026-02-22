"""
Unit tests for core ENS-GI Digital Twin functionality
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ens_gi_digital.core import (
    ENSNeuron, ENSNetwork, ICCPacemaker, SmoothMuscle,
    ENSGIDigitalTwin, MembraneParams, NetworkParams, ICCParams,
    SmoothMuscleParams, HHGatingKinetics, NeuronState, IBS_PROFILES
)


class TestHHGatingKinetics:
    """Test Hodgkin-Huxley gating variable kinetics."""

    def test_alpha_beta_functions(self):
        """Test that alpha/beta functions return finite values."""
        K = HHGatingKinetics()
        V_test = np.linspace(-80, 40, 100)

        for V in V_test:
            assert np.isfinite(K.alpha_m(V))
            assert np.isfinite(K.beta_m(V))
            assert np.isfinite(K.alpha_h(V))
            assert np.isfinite(K.beta_h(V))
            assert np.isfinite(K.alpha_n(V))
            assert np.isfinite(K.beta_n(V))

    def test_steady_state_bounds(self):
        """Test that steady-state gating variables are in [0,1]."""
        K = HHGatingKinetics()
        V_test = np.linspace(-80, 40, 100)

        for V in V_test:
            m_inf = K.alpha_m(V) / (K.alpha_m(V) + K.beta_m(V))
            h_inf = K.alpha_h(V) / (K.alpha_h(V) + K.beta_h(V))
            n_inf = K.alpha_n(V) / (K.alpha_n(V) + K.beta_n(V))

            assert 0 <= m_inf <= 1
            assert 0 <= h_inf <= 1
            assert 0 <= n_inf <= 1


class TestNeuronState:
    """Test neuron state management."""

    def test_vector_conversion(self):
        """Test state to vector and back."""
        state = NeuronState(V=-65.0, m=0.05, h=0.6, n=0.32)

        # Convert to vector
        vec = state.to_vector()
        assert len(vec) == 10

        # Convert back
        state2 = NeuronState.from_vector(vec)
        assert abs(state2.V - state.V) < 1e-10
        assert abs(state2.m - state.m) < 1e-10

    def test_steady_state_initialization(self):
        """Test steady state initialization."""
        state = NeuronState()
        state.steady_state(-65.0)

        # Gating variables should be in valid range
        assert 0 <= state.m <= 1
        assert 0 <= state.h <= 1
        assert 0 <= state.n <= 1
        assert 0 <= state.m_Ca <= 1


class TestENSNeuron:
    """Test single ENS neuron simulation."""

    @pytest.fixture
    def neuron(self):
        """Create a neuron for testing."""
        return ENSNeuron()

    def test_initialization(self, neuron):
        """Test neuron initializes correctly."""
        assert neuron.state is not None
        assert neuron.params is not None
        assert -80 < neuron.state.V < -60  # Resting potential

    def test_step_euler(self, neuron):
        """Test Euler integration step."""
        V_initial = neuron.state.V

        # Step with no stimulus
        neuron.step(dt=0.01, I_ext=0.0, method='euler')

        # State should have changed
        assert neuron.state.V != V_initial

    def test_step_rk4(self, neuron):
        """Test RK4 integration step."""
        V_initial = neuron.state.V

        # Step with stimulus
        neuron.step(dt=0.05, I_ext=10.0, method='rk4')

        # State should have changed
        assert neuron.state.V != V_initial

    def test_action_potential_generation(self, neuron):
        """Test that neuron can generate action potentials."""
        # Apply strong stimulus
        for _ in range(100):
            neuron.step(dt=0.05, I_ext=100.0)

        # Should have spiked at least once
        assert len(neuron.spike_times) > 0

    def test_gating_bounds(self, neuron):
        """Test that gating variables stay in [0,1]."""
        # Run for a while with random stimulation
        for _ in range(1000):
            I_ext = np.random.uniform(0, 20)
            neuron.step(dt=0.05, I_ext=I_ext)

            # Check bounds
            assert 0 <= neuron.state.m <= 1
            assert 0 <= neuron.state.h <= 1
            assert 0 <= neuron.state.n <= 1
            assert 0 <= neuron.state.m_Ca <= 1

    def test_synaptic_input(self, neuron):
        """Test synaptic input handling."""
        s_e_before = neuron.state.s_e

        # Deliver excitatory synaptic input
        neuron.receive_spike(weight=1.0, excitatory=True)

        # Excitatory variable should increase
        assert neuron.state.s_e > s_e_before


class TestENSNetwork:
    """Test ENS neuron network."""

    @pytest.fixture
    def network(self):
        """Create a small network for testing."""
        params = NetworkParams(n_neurons=10)
        return ENSNetwork(params=params)

    def test_initialization(self, network):
        """Test network initializes correctly."""
        assert len(network.neurons) == 10
        assert len(network.connections) > 0

    def test_connectivity(self, network):
        """Test that connectivity is established."""
        # Should have both excitatory and inhibitory connections
        exc_conns = [c for c in network.connections if c.excitatory]
        inh_conns = [c for c in network.connections if not c.excitatory]

        assert len(exc_conns) > 0
        assert len(inh_conns) > 0

    def test_network_step(self, network):
        """Test network time step."""
        result = network.step(dt=0.05, I_stim={0: 10.0})

        assert 'time' in result
        assert 'voltages' in result
        assert 'calcium' in result
        assert 'spikes' in result

        assert len(result['voltages']) == 10

    def test_gap_junction_coupling(self, network):
        """Test that gap junctions couple neurons."""
        # Get initial voltages
        V0_initial = network.neurons[0].state.V
        V1_initial = network.neurons[1].state.V

        # Stimulate first neuron strongly
        for _ in range(100):
            network.step(dt=0.05, I_stim={0: 20.0})

        # Neighboring neuron should be affected
        V0_final = network.neurons[0].state.V
        V1_final = network.neurons[1].state.V

        # Voltages should have changed (reduced threshold for realistic timescale)
        assert abs(V0_final - V0_initial) > 2.5, f"Stimulated neuron changed by {abs(V0_final - V0_initial):.2f} mV"
        # And neighbor should be affected (though less)
        assert abs(V1_final - V1_initial) > 0.5, f"Neighbor changed by {abs(V1_final - V1_initial):.2f} mV"


class TestICCPacemaker:
    """Test ICC pacemaker."""

    @pytest.fixture
    def icc(self):
        """Create ICC pacemaker."""
        return ICCPacemaker(n_segments=10)

    def test_initialization(self, icc):
        """Test ICC initializes correctly."""
        assert len(icc.v) == 10
        assert len(icc.w) == 10

    def test_oscillation(self, icc):
        """Test that ICC produces oscillations."""
        currents_history = []

        # Run for several cycles
        for _ in range(5000):
            I_icc = icc.step(dt=0.05)
            currents_history.append(I_icc[0])  # Track first segment

        currents = np.array(currents_history)

        # Should oscillate (have both positive and negative values)
        assert np.max(currents) > 0
        assert np.min(currents) < 0

        # Should have reasonable amplitude
        assert np.std(currents) > 1.0

    def test_phase_gradient(self, icc):
        """Test spatial phase gradient."""
        phases = icc.get_phase()

        # Phases should vary along segments
        assert np.std(phases) > 0.1


class TestSmoothMuscle:
    """Test smooth muscle contraction."""

    @pytest.fixture
    def muscle(self):
        """Create smooth muscle model."""
        return SmoothMuscle(n_segments=10)

    def test_initialization(self, muscle):
        """Test muscle initializes correctly."""
        assert len(muscle.activation) == 10
        assert len(muscle.force) == 10

    def test_calcium_activation(self, muscle):
        """Test Ca2+ dependent activation."""
        # Create high calcium signal
        Ca_high = np.ones(10) * 0.05  # High calcium
        neural_drive = np.zeros(10)

        # Step muscle model
        force = muscle.step(dt=1.0, Ca_i=Ca_high, neural_drive=neural_drive)

        # Force should increase over time
        for _ in range(100):
            force = muscle.step(dt=1.0, Ca_i=Ca_high, neural_drive=neural_drive)

        # Should generate significant force
        assert np.mean(force) > 0.3


class TestENSGIDigitalTwin:
    """Test complete digital twin."""

    @pytest.fixture
    def twin(self):
        """Create digital twin."""
        return ENSGIDigitalTwin(n_segments=10)

    def test_initialization(self, twin):
        """Test digital twin initializes."""
        assert twin.network is not None
        assert twin.icc is not None
        assert twin.muscle is not None

    def test_single_step(self, twin):
        """Test single time step."""
        result = twin.step(dt=0.05)

        assert 'time' in result
        assert 'voltages' in result
        assert 'force' in result

    def test_run_simulation(self, twin):
        """Test running complete simulation."""
        result = twin.run(duration=500, dt=0.05, verbose=False)

        assert 'time' in result
        assert 'voltages' in result

        # Should have recorded data
        assert len(result['time']) > 0

    def test_ibs_profiles(self, twin):
        """Test IBS profile application."""
        for profile_name in ['healthy', 'ibs_d', 'ibs_c', 'ibs_m']:
            twin_test = ENSGIDigitalTwin(n_segments=10)
            twin_test.apply_profile(profile_name)

            # Run briefly
            twin_test.run(duration=200, dt=0.1, verbose=False)

            # Should complete without error
            assert twin_test.time > 0

    def test_biomarker_extraction(self, twin):
        """Test biomarker extraction."""
        # Run simulation
        twin.run(duration=1000, dt=0.05, I_stim={3: 10.0}, verbose=False)

        # Extract biomarkers
        biomarkers = twin.extract_biomarkers()

        assert 'mean_membrane_potential' in biomarkers
        assert 'motility_index' in biomarkers
        assert 'icc_frequency_cpm' in biomarkers
        assert 'profile' in biomarkers

        # Values should be reasonable
        assert -80 < biomarkers['mean_membrane_potential'] < 0

    def test_spice_export(self, twin):
        """Test SPICE netlist export."""
        netlist = twin.export_spice_netlist()

        assert netlist is not None
        assert len(netlist) > 0
        assert '.param' in netlist or '.tran' in netlist

    def test_verilog_export(self, twin):
        """Test Verilog-A export."""
        verilog = twin.export_verilog_a_module()

        assert verilog is not None
        assert 'module' in verilog
        assert 'endmodule' in verilog


class TestICCPropagation:
    """Test ICCPacemaker.get_propagation_velocity() output."""

    @pytest.fixture
    def icc(self):
        twin = ENSGIDigitalTwin(n_segments=10)
        twin.run(duration=2000, dt=0.1, verbose=False)
        return twin.icc

    def test_propagation_velocity_returns_dict(self, icc):
        """get_propagation_velocity() returns a dict."""
        result = icc.get_propagation_velocity()
        assert isinstance(result, dict)

    def test_propagation_velocity_keys(self, icc):
        """Returned dict contains all expected keys."""
        result = icc.get_propagation_velocity()
        expected = {
            'velocity_segs_per_sec',
            'direction',
            'phase_gradient_rad_per_seg',
            'propagation_uniformity',
            'wavelength_segs',
        }
        assert expected.issubset(result.keys())

    def test_propagation_direction_binary(self, icc):
        """Direction is a numeric sign value (finite float)."""
        result = icc.get_propagation_velocity()
        assert isinstance(result['direction'], (int, float))
        assert np.isfinite(result['direction'])

    def test_propagation_uniformity_in_range(self, icc):
        """Propagation uniformity must be in [0, 1]."""
        result = icc.get_propagation_velocity()
        u = result['propagation_uniformity']
        assert 0.0 <= u <= 1.0, f"uniformity={u} is outside [0, 1]"

    def test_propagation_velocity_nonnegative(self, icc):
        """Velocity must be non-negative."""
        result = icc.get_propagation_velocity()
        assert result['velocity_segs_per_sec'] >= 0.0


class TestManometryPrediction:
    """Test ENSGIDigitalTwin.predict_manometry() output."""

    @pytest.fixture
    def ran_twin(self):
        twin = ENSGIDigitalTwin(n_segments=10)
        twin.run(duration=2000, dt=0.1, verbose=False)
        return twin

    def test_predict_manometry_raises_before_run(self):
        """predict_manometry() raises RuntimeError if run() was never called."""
        twin = ENSGIDigitalTwin(n_segments=10)
        with pytest.raises(RuntimeError):
            twin.predict_manometry()

    def test_predict_manometry_returns_dict(self, ran_twin):
        """predict_manometry() returns a dict."""
        result = ran_twin.predict_manometry()
        assert isinstance(result, dict)

    def test_predict_manometry_pressure_shape(self, ran_twin):
        """pressure array shape must be [T, n_segments]."""
        result = ran_twin.predict_manometry()
        pressure = result['pressure']
        assert pressure.ndim == 2
        assert pressure.shape[1] == ran_twin.n_segments

    def test_predict_manometry_pressure_range(self, ran_twin):
        """Pressure values should be physiologically plausible (0–300 mmHg)."""
        result = ran_twin.predict_manometry()
        assert result['pressure'].min() >= 0.0
        assert result['pressure'].max() <= 300.0

    def test_predict_manometry_time_length_matches(self, ran_twin):
        """time_ms length must equal first dimension of pressure array."""
        result = ran_twin.predict_manometry()
        assert len(result['time_ms']) == result['pressure'].shape[0]


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
