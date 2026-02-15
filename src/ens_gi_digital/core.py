"""
ENS-GI Digital Twin — Core Simulation Engine
=============================================
A multiscale, physics-based ENS (Enteric Nervous System) - GI (Gastrointestinal)
simulator serving as the foundation for:
  1. Biological research simulator
  2. Neuromorphic / hardware-inspired model
  3. Clinically predictive system

Architecture:
  Layer 1: Cellular Electrophysiology (HH + Ca²⁺ + synaptic)
  Layer 2: Network & Propagation (coupled ENS, E/I balance, reflex loops)
  Layer 3: ICC Pacemaker & Motility (slow waves, Ca²⁺ dynamics, smooth muscle)

Author: Mahdad
License: MIT
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import json


# ═══════════════════════════════════════════════════════════════
# Layer 1: Cellular Electrophysiology
# ═══════════════════════════════════════════════════════════════

@dataclass
class MembraneParams:
    """Biophysical parameters for ENS neuron membrane model.
    
    Based on Hodgkin-Huxley formalism extended with:
    - L-type Ca²⁺ channel (enteric neuron specific)
    - Synaptic conductances (excitatory/inhibitory)
    - Neurotransmitter modulation factors
    """
    # Membrane capacitance (μF/cm²)
    C_m: float = 1.0
    
    # Ion channel maximal conductances (mS/cm²)
    g_Na: float = 120.0    # Fast Na⁺ (action potential upstroke)
    g_K: float = 36.0      # Delayed rectifier K⁺
    g_L: float = 0.3       # Leak
    g_Ca: float = 4.0      # L-type Ca²⁺ (enteric-specific)
    g_KCa: float = 5.0     # Ca²⁺-activated K⁺ (AHP)
    g_A: float = 8.0       # A-type K⁺ (transient outward)
    
    # Reversal potentials (mV)
    E_Na: float = 50.0
    E_K: float = -77.0
    E_L: float = -54.4
    E_Ca: float = 120.0
    
    # Synaptic parameters
    g_syn_e: float = 0.5   # Excitatory synaptic conductance (mS/cm²)
    g_syn_i: float = 1.0   # Inhibitory synaptic conductance
    E_syn_e: float = 0.0   # Excitatory reversal (ACh/substance P)
    E_syn_i: float = -80.0 # Inhibitory reversal (NO/VIP)
    tau_syn_e: float = 5.0 # Excitatory time constant (ms)
    tau_syn_i: float = 10.0 # Inhibitory time constant (ms)
    
    # Calcium dynamics
    tau_Ca: float = 50.0   # Ca²⁺ clearance time constant (ms) — faster clearance
    k_Ca: float = 0.0002   # Ca²⁺ influx coupling factor (reduced)
    Ca_half: float = 0.001 # Half-activation for KCa (mM)
    
    # Neurotransmitter modulation
    serotonin_factor: float = 1.0  # 5-HT modulation (↑ excitability)
    ach_factor: float = 1.0         # ACh modulation
    no_factor: float = 1.0          # NO modulation (↓ excitability)


class HHGatingKinetics:
    """Hodgkin-Huxley style gating variable kinetics.
    
    Extended for enteric neuron-specific channels including
    L-type Ca²⁺, KCa, and A-type K⁺.
    
    SPICE equivalence: Each gate maps to a nonlinear conductance
    controlled by voltage-dependent state variables.
    """
    
    @staticmethod
    def alpha_m(V: float) -> float:
        """Na⁺ activation rate."""
        dV = V + 40.0
        if abs(dV) < 1e-7:
            return 1.0
        return 0.1 * dV / (1.0 - np.exp(-dV / 10.0))
    
    @staticmethod
    def beta_m(V: float) -> float:
        return 4.0 * np.exp(-(V + 65.0) / 18.0)
    
    @staticmethod
    def alpha_h(V: float) -> float:
        """Na⁺ inactivation rate."""
        return 0.07 * np.exp(-(V + 65.0) / 20.0)
    
    @staticmethod
    def beta_h(V: float) -> float:
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
    
    @staticmethod
    def alpha_n(V: float) -> float:
        """K⁺ activation rate."""
        dV = V + 55.0
        if abs(dV) < 1e-7:
            return 0.1
        return 0.01 * dV / (1.0 - np.exp(-dV / 10.0))
    
    @staticmethod
    def beta_n(V: float) -> float:
        return 0.125 * np.exp(-(V + 65.0) / 80.0)
    
    @staticmethod
    def m_Ca_inf(V: float) -> float:
        """L-type Ca²⁺ steady-state activation (enteric-specific)."""
        return 1.0 / (1.0 + np.exp(-(V + 20.0) / 9.0))
    
    @staticmethod
    def tau_m_Ca(V: float) -> float:
        """L-type Ca²⁺ activation time constant."""
        return 1.0 + 10.0 / (1.0 + np.exp((V + 20.0) / 9.0))
    
    @staticmethod
    def a_inf(V: float) -> float:
        """A-type K⁺ steady-state activation."""
        return 1.0 / (1.0 + np.exp(-(V + 45.0) / 14.5))
    
    @staticmethod
    def b_inf(V: float) -> float:
        """A-type K⁺ steady-state inactivation."""
        return 1.0 / (1.0 + np.exp((V + 70.0) / 7.5))
    
    @staticmethod
    def KCa_inf(Ca: float, Ca_half: float = 0.001) -> float:
        """Ca²⁺-activated K⁺ channel activation."""
        return Ca / (Ca + Ca_half)


@dataclass
class NeuronState:
    """Complete state vector for a single ENS neuron.
    
    Maps to SPICE node voltages:
      V    → main membrane node
      m,h  → Na⁺ gate control voltages
      n    → K⁺ gate control voltage
      m_Ca → Ca²⁺ gate control voltage
      s_e, s_i → synaptic conductance nodes
      Ca_i → intracellular Ca²⁺ concentration node
    """
    V: float = -65.0       # Membrane potential (mV)
    m: float = 0.05        # Na⁺ activation
    h: float = 0.6         # Na⁺ inactivation
    n: float = 0.32        # K⁺ activation
    m_Ca: float = 0.0      # Ca²⁺ activation
    a: float = 0.0         # A-type K⁺ activation
    b: float = 1.0         # A-type K⁺ inactivation
    s_e: float = 0.0       # Excitatory synaptic variable
    s_i: float = 0.0       # Inhibitory synaptic variable
    Ca_i: float = 0.0001   # Intracellular [Ca²⁺] (mM)
    
    def to_vector(self) -> np.ndarray:
        return np.array([self.V, self.m, self.h, self.n, self.m_Ca,
                         self.a, self.b, self.s_e, self.s_i, self.Ca_i])
    
    @classmethod
    def from_vector(cls, vec: np.ndarray) -> 'NeuronState':
        return cls(V=vec[0], m=vec[1], h=vec[2], n=vec[3], m_Ca=vec[4],
                   a=vec[5], b=vec[6], s_e=vec[7], s_i=vec[8], Ca_i=vec[9])
    
    def steady_state(self, V: Optional[float] = None):
        """Initialize to steady state at given voltage."""
        if V is not None:
            self.V = V
        K = HHGatingKinetics
        self.m = K.alpha_m(self.V) / (K.alpha_m(self.V) + K.beta_m(self.V))
        self.h = K.alpha_h(self.V) / (K.alpha_h(self.V) + K.beta_h(self.V))
        self.n = K.alpha_n(self.V) / (K.alpha_n(self.V) + K.beta_n(self.V))
        self.m_Ca = K.m_Ca_inf(self.V)
        self.a = K.a_inf(self.V)
        self.b = K.b_inf(self.V)


class ENSNeuron:
    """Single ENS (Enteric Nervous System) neuron model.
    
    Implements extended Hodgkin-Huxley formalism with:
    - Fast Na⁺, delayed rectifier K⁺ (standard HH)
    - L-type Ca²⁺ channel (enteric neuron enriched)
    - Ca²⁺-activated K⁺ (afterhyperpolarization)
    - A-type K⁺ (transient outward)
    - Excitatory/inhibitory synaptic inputs
    - Intracellular Ca²⁺ dynamics
    
    Governing equation:
        C_m dV/dt = -I_Na - I_K - I_L - I_Ca - I_KCa - I_A - I_syn + I_ext
    
    SPICE mapping:
        C_m → capacitor
        Each I_ion → nonlinear voltage-controlled current source
        Gating variables → auxiliary RC networks with nonlinear R(V)
    
    Verilog-A mapping:
        Each current → analog function block
        ddt() for time derivatives
        V(node) for voltage access
    """
    
    def __init__(self, params: Optional[MembraneParams] = None, neuron_id: int = 0):
        self.params = params or MembraneParams()
        self.state = NeuronState()
        self.state.steady_state(-65.0 + np.random.uniform(-3, 3))
        self.id = neuron_id
        self.K = HHGatingKinetics()
        
        # Recording
        self.spike_times: List[float] = []
        self._prev_V = self.state.V
    
    def compute_currents(self, state: NeuronState, 
                          I_ext: float = 0.0) -> Dict[str, float]:
        """Compute all ionic and synaptic currents.
        
        Returns dict of current components for analysis/recording.
        Each current follows the convention I = g * (V - E).
        """
        p = self.params
        s = state
        
        # Neurotransmitter modulation
        g_Na_mod = p.g_Na * p.serotonin_factor
        g_K_mod = p.g_K * p.no_factor
        
        currents = {
            'I_Na':  g_Na_mod * s.m**3 * s.h * (s.V - p.E_Na),
            'I_K':   g_K_mod * s.n**4 * (s.V - p.E_K),
            'I_L':   p.g_L * (s.V - p.E_L),
            'I_Ca':  p.g_Ca * s.m_Ca * (s.V - p.E_Ca),
            'I_KCa': p.g_KCa * self.K.KCa_inf(s.Ca_i, p.Ca_half) * (s.V - p.E_K),
            'I_A':   p.g_A * s.a * s.b * (s.V - p.E_K),
            'I_syn_e': p.g_syn_e * s.s_e * (s.V - p.E_syn_e),
            'I_syn_i': p.g_syn_i * s.s_i * (s.V - p.E_syn_i),
            'I_ext': I_ext,
        }
        return currents
    
    def derivatives(self, state: NeuronState, 
                     I_ext: float = 0.0,
                     I_couple: float = 0.0,
                     I_icc: float = 0.0) -> np.ndarray:
        """Compute state derivatives (RHS of ODE system).
        
        This is the core mathematical model that maps identically to:
        - SPICE nodal equations
        - Verilog-A analog blocks
        - Clinical parameter fitting targets
        
        Returns: d/dt [V, m, h, n, m_Ca, a, b, s_e, s_i, Ca_i]
        """
        p = self.params
        s = state
        K = self.K
        
        currents = self.compute_currents(state, I_ext)
        
        # Membrane voltage (main equation)
        I_ion_total = sum(v for k, v in currents.items() if k != 'I_ext')
        dVdt = (-I_ion_total + I_ext + I_couple + I_icc) / p.C_m
        
        # HH gating variables
        dmdt = K.alpha_m(s.V) * (1 - s.m) - K.beta_m(s.V) * s.m
        dhdt = K.alpha_h(s.V) * (1 - s.h) - K.beta_h(s.V) * s.h
        dndt = K.alpha_n(s.V) * (1 - s.n) - K.beta_n(s.V) * s.n
        
        # Ca²⁺ channel gating
        dm_Ca_dt = (K.m_Ca_inf(s.V) - s.m_Ca) / K.tau_m_Ca(s.V)
        
        # A-type K⁺ gating (fast dynamics)
        tau_a = 5.0  # ms
        tau_b = 50.0  # ms
        dadt = (K.a_inf(s.V) - s.a) / tau_a
        dbdt = (K.b_inf(s.V) - s.b) / tau_b
        
        # Synaptic dynamics (exponential decay)
        ds_e_dt = -s.s_e / p.tau_syn_e
        ds_i_dt = -s.s_i / p.tau_syn_i
        
        # Intracellular Ca²⁺ dynamics
        # Ca²⁺ entry through L-type channels, clearance by pumps/buffers
        I_Ca = currents['I_Ca']
        dCa_dt = -p.k_Ca * I_Ca - s.Ca_i / p.tau_Ca
        
        return np.array([dVdt, dmdt, dhdt, dndt, dm_Ca_dt,
                         dadt, dbdt, ds_e_dt, ds_i_dt, dCa_dt])
    
    def step(self, dt: float, I_ext: float = 0.0, 
             I_couple: float = 0.0, I_icc: float = 0.0,
             method: str = 'rk4') -> None:
        """Advance neuron state by dt milliseconds.
        
        Methods:
            'euler': Forward Euler (fastest, least accurate)
            'rk2':   Midpoint method
            'rk4':   Classical Runge-Kutta (recommended)
        """
        y = self.state.to_vector()
        
        if method == 'euler':
            dy = self.derivatives(NeuronState.from_vector(y), I_ext, I_couple, I_icc)
            y_new = y + dt * dy
            
        elif method == 'rk2':
            k1 = self.derivatives(NeuronState.from_vector(y), I_ext, I_couple, I_icc)
            k2 = self.derivatives(NeuronState.from_vector(y + 0.5*dt*k1), I_ext, I_couple, I_icc)
            y_new = y + dt * k2
            
        elif method == 'rk4':
            k1 = self.derivatives(NeuronState.from_vector(y), I_ext, I_couple, I_icc)
            k2 = self.derivatives(NeuronState.from_vector(y + 0.5*dt*k1), I_ext, I_couple, I_icc)
            k3 = self.derivatives(NeuronState.from_vector(y + 0.5*dt*k2), I_ext, I_couple, I_icc)
            k4 = self.derivatives(NeuronState.from_vector(y + dt*k3), I_ext, I_couple, I_icc)
            y_new = y + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Enforce bounds
        y_new[1:7] = np.clip(y_new[1:7], 0, 1)  # gating variables [0,1]
        y_new[7:9] = np.maximum(y_new[7:9], 0)    # synaptic variables ≥ 0
        y_new[9] = max(y_new[9], 0)                # Ca²⁺ ≥ 0
        
        self.state = NeuronState.from_vector(y_new)
        
        # Spike detection (threshold crossing)
        if self._prev_V <= 0 and self.state.V > 0:
            self.spike_times.append(0)  # caller should set actual time
        self._prev_V = self.state.V
    
    def receive_spike(self, weight: float = 1.0, excitatory: bool = True):
        """Process incoming synaptic event."""
        if excitatory:
            self.state.s_e += weight * self.params.ach_factor
        else:
            self.state.s_i += weight * self.params.no_factor


# ═══════════════════════════════════════════════════════════════
# Layer 2: Network & Propagation
# ═══════════════════════════════════════════════════════════════

@dataclass
class NetworkParams:
    """Parameters for ENS network topology and coupling."""
    n_neurons: int = 20
    coupling_strength: float = 0.3     # Gap junction conductance (mS/cm²)
    propagation_delay: float = 1.0     # Axonal delay per segment (ms)
    excitatory_weight: float = 0.5     # Synaptic weight (excitatory)
    inhibitory_weight: float = 0.3     # Synaptic weight (inhibitory)
    excitatory_radius: int = 2         # Connectivity radius (forward)
    inhibitory_radius: int = 3         # Connectivity radius (backward)
    reflex_gain: float = 0.5           # Peristaltic reflex gain


@dataclass
class SynapticConnection:
    """Single synaptic connection between neurons."""
    pre_id: int
    post_id: int
    weight: float
    excitatory: bool
    delay: float = 0.0  # ms


class ENSNetwork:
    """Coupled network of ENS neurons with peristaltic reflex architecture.
    
    Implements:
    - Linear chain topology (gut longitudinal axis)
    - Gap junction (electrical) coupling
    - Chemical synapses with E/I balance
    - Ascending excitation / descending inhibition (Bayliss-Starling reflex)
    - Wave propagation dynamics
    
    Network layout:
        [N0] ---gap--- [N1] ---gap--- [N2] --- ... --- [Nn]
              ←exc→          ←exc→
              ←──────inh──────→
    
    SPICE mapping: Neurons = subcircuits, gap junctions = resistors,
                   synapses = voltage-controlled current sources with delay
    """
    
    def __init__(self, params: Optional[NetworkParams] = None,
                 neuron_params: Optional[MembraneParams] = None):
        self.params = params or NetworkParams()
        self.neuron_params = neuron_params or MembraneParams()
        
        # Create neurons
        self.neurons = [
            ENSNeuron(MembraneParams(**{**self.neuron_params.__dict__}), i)
            for i in range(self.params.n_neurons)
        ]
        
        # Build connectivity
        self.connections = self._build_connections()
        
        # Spike buffer for delayed transmission
        self._spike_buffer: List[Tuple[float, int, float, bool]] = []
        
        # Time tracking
        self.time = 0.0
    
    def _build_connections(self) -> List[SynapticConnection]:
        """Build peristaltic reflex connectivity.
        
        Ascending excitation: oral (backward) neurons excite
        Descending inhibition: anal (forward) neurons inhibit
        This creates the Bayliss-Starling reflex arc.
        """
        connections = []
        N = self.params.n_neurons
        p = self.params
        
        for i in range(N):
            # Ascending excitation (backward connections)
            for j in range(max(0, i - p.excitatory_radius), i):
                connections.append(SynapticConnection(
                    pre_id=i, post_id=j,
                    weight=p.excitatory_weight * (1 - abs(i-j)/(p.excitatory_radius+1)),
                    excitatory=True,
                    delay=p.propagation_delay * abs(i-j)
                ))
            
            # Descending inhibition (forward connections)
            for j in range(i+1, min(N, i + p.inhibitory_radius + 1)):
                connections.append(SynapticConnection(
                    pre_id=i, post_id=j,
                    weight=p.inhibitory_weight * (1 - abs(i-j)/(p.inhibitory_radius+1)),
                    excitatory=False,
                    delay=p.propagation_delay * abs(i-j)
                ))
        
        return connections
    
    def step(self, dt: float, 
             I_stim: Optional[Dict[int, float]] = None,
             I_icc: Optional[np.ndarray] = None,
             method: str = 'rk4') -> Dict:
        """Advance entire network by dt.
        
        Args:
            dt: Time step (ms)
            I_stim: External stimulation {neuron_id: current}
            I_icc: ICC pacemaker current array (one per neuron)
            method: Integration method
            
        Returns:
            Dict with network state snapshot
        """
        I_stim = I_stim or {}
        N = self.params.n_neurons
        
        # Process delayed spikes
        new_buffer = []
        for (fire_time, post_id, weight, exc) in self._spike_buffer:
            if self.time >= fire_time:
                self.neurons[post_id].receive_spike(weight, exc)
            else:
                new_buffer.append((fire_time, post_id, weight, exc))
        self._spike_buffer = new_buffer
        
        # Compute gap junction coupling currents
        coupling_currents = np.zeros(N)
        g_gap = self.params.coupling_strength
        for i in range(N):
            if i > 0:
                coupling_currents[i] += g_gap * (self.neurons[i-1].state.V - self.neurons[i].state.V)
            if i < N - 1:
                coupling_currents[i] += g_gap * (self.neurons[i+1].state.V - self.neurons[i].state.V)
        
        # Step each neuron
        spikes = []
        for i in range(N):
            I_ext = I_stim.get(i, 0.0)
            I_icc_i = I_icc[i] if I_icc is not None else 0.0
            
            prev_V = self.neurons[i].state.V
            self.neurons[i].step(dt, I_ext, coupling_currents[i], I_icc_i, method)
            
            # Detect spikes
            if prev_V <= 0 and self.neurons[i].state.V > 0:
                spikes.append(i)
                self.neurons[i].spike_times.append(self.time)
                
                # Queue synaptic transmission
                for conn in self.connections:
                    if conn.pre_id == i:
                        self._spike_buffer.append((
                            self.time + conn.delay,
                            conn.post_id,
                            conn.weight,
                            conn.excitatory
                        ))
        
        self.time += dt
        
        return {
            'time': self.time,
            'voltages': np.array([n.state.V for n in self.neurons]),
            'calcium': np.array([n.state.Ca_i for n in self.neurons]),
            'spikes': spikes,
        }
    
    def get_state_matrix(self) -> np.ndarray:
        """Get full state matrix [N_neurons × N_state_vars]."""
        return np.array([n.state.to_vector() for n in self.neurons])


# ═══════════════════════════════════════════════════════════════
# Layer 3: ICC Pacemaker & Motility
# ═══════════════════════════════════════════════════════════════

@dataclass
class ICCParams:
    """Interstitial Cells of Cajal (ICC) pacemaker parameters.
    
    ICCs generate the electrical slow waves that pace GI motility.
    Model based on coupled oscillator framework.
    
    Note on timescales: physiological slow waves are ~3 cpm (0.05 Hz).
    omega is in rad/ms; multiply by (1000*60)/(2π) to get cpm.
    For ~3 cpm: omega ≈ 0.000314 rad/ms (20s period)
    For demo/accelerated: omega ≈ 0.005 rad/ms (1.26s period ≈ ~48 cpm)
    """
    omega: float = 0.000314    # Angular frequency (rad/ms) → ~3 cpm (realistic)
    amplitude: float = 12.0    # Slow wave amplitude (mV) — strong enough to modulate spiking
    phase_gradient: float = 0.3 # Phase lag per segment (rad)
    coupling_icc: float = 0.1  # ICC-to-ICC coupling
    
    # Nonlinear oscillator (FitzHugh-Nagumo based)
    epsilon: float = 0.08      # Time scale separation
    a_fhn: float = 0.7         # FHN parameter
    b_fhn: float = 0.8         # FHN parameter


@dataclass
class SmoothMuscleParams:
    """Smooth muscle contraction parameters.
    
    Electromechanical coupling: electrical activity → Ca²⁺ → force
    """
    k_activation: float = 0.01   # E→M coupling gain
    tau_activation: float = 100.0 # Activation time constant (ms)
    tau_relaxation: float = 300.0 # Relaxation time constant (ms)
    max_force: float = 1.0       # Normalized maximum force
    resting_tone: float = 0.05   # Baseline tone
    hill_n: float = 3.0          # Hill coefficient for Ca²⁺→force
    Ca_half_force: float = 0.02  # Ca²⁺ for half-maximal force (mM) — matched to model Ca range


class ICCPacemaker:
    """ICC pacemaker network generating slow waves.
    
    Uses FitzHugh-Nagumo oscillators with phase gradient
    to produce propagating slow waves along the gut.
    
    Equations:
        dv/dt = v - v³/3 - w + I_ext
        dw/dt = ε(v + a - bw)
    
    SPICE mapping: Wien bridge oscillator or ring oscillator
    Verilog-A: Behavioral oscillator module
    """
    
    def __init__(self, n_segments: int = 20, 
                 params: Optional[ICCParams] = None):
        self.params = params or ICCParams()
        self.n = n_segments
        
        # FHN state: v (fast), w (slow) per segment
        self.v = np.zeros(n_segments)
        self.w = np.zeros(n_segments)
        
        # Initialize with phase gradient
        for i in range(n_segments):
            phase = i * self.params.phase_gradient
            self.v[i] = np.cos(phase) * 0.5
            self.w[i] = np.sin(phase) * 0.3
        
        self.time = 0.0
    
    def step(self, dt: float) -> np.ndarray:
        """Advance ICC oscillators and return slow wave current.
        
        Returns: Array of ICC-generated currents (one per segment)
        """
        p = self.params
        
        # FHN dynamics with nearest-neighbor coupling
        dv = self.v - self.v**3 / 3.0 - self.w
        dw = p.epsilon * (self.v + p.a_fhn - p.b_fhn * self.w)
        
        # ICC-to-ICC diffusive coupling
        coupling = np.zeros(self.n)
        coupling[1:] += p.coupling_icc * (self.v[:-1] - self.v[1:])
        coupling[:-1] += p.coupling_icc * (self.v[1:] - self.v[:-1])
        dv += coupling
        
        self.v += dv * dt
        self.w += dw * dt
        self.time += dt
        
        # Convert to current (scale to physiological range)
        return p.amplitude * self.v
    
    def get_phase(self) -> np.ndarray:
        """Compute instantaneous phase of each oscillator."""
        return np.arctan2(self.w, self.v)
    
    def get_frequency(self) -> float:
        """Estimate dominant frequency (cpm)."""
        return self.params.omega / (2 * np.pi) * 1000 * 60  # rad/ms → cpm


class SmoothMuscle:
    """Smooth muscle contraction model.
    
    Converts electrical/chemical signals to mechanical force.
    
    Pathway: ENS activity → Ca²⁺ → cross-bridge cycling → force
    
    SPICE mapping: E→M transducer (voltage-to-force converter)
    """
    
    def __init__(self, n_segments: int = 20,
                 params: Optional[SmoothMuscleParams] = None):
        self.params = params or SmoothMuscleParams()
        self.n = n_segments
        
        # State variables
        self.activation = np.ones(n_segments) * self.params.resting_tone
        self.force = np.ones(n_segments) * self.params.resting_tone
    
    def step(self, dt: float, 
             Ca_i: np.ndarray, 
             neural_drive: np.ndarray) -> np.ndarray:
        """Compute contractile force from Ca²⁺ and neural input.
        
        Args:
            dt: Time step (ms)
            Ca_i: Intracellular Ca²⁺ per segment
            neural_drive: Net neural excitation per segment
            
        Returns: Force array (normalized 0-1)
        """
        p = self.params
        
        # Ca²⁺ → activation (Hill function)
        Ca_activation = Ca_i**p.hill_n / (Ca_i**p.hill_n + p.Ca_half_force**p.hill_n)
        
        # Combined drive
        total_drive = Ca_activation + p.k_activation * neural_drive
        target_activation = np.clip(total_drive + p.resting_tone, 0, 1)
        
        # First-order dynamics with asymmetric time constants
        for i in range(self.n):
            if target_activation[i] > self.activation[i]:
                tau = p.tau_activation
            else:
                tau = p.tau_relaxation
            self.activation[i] += (target_activation[i] - self.activation[i]) * dt / tau
        
        self.force = self.activation * p.max_force
        return self.force


# ═══════════════════════════════════════════════════════════════
# Unified Digital Twin
# ═══════════════════════════════════════════════════════════════

@dataclass
class IBSProfile:
    """IBS patient parameter profile for clinical simulation."""
    name: str
    membrane_mods: Dict = field(default_factory=dict)
    network_mods: Dict = field(default_factory=dict)
    icc_mods: Dict = field(default_factory=dict)
    muscle_mods: Dict = field(default_factory=dict)


# Predefined IBS profiles
IBS_PROFILES = {
    'healthy': IBSProfile(name='Healthy Control'),
    
    'ibs_d': IBSProfile(
        name='IBS-D (Diarrhea-predominant)',
        membrane_mods={'g_Na': 150.0, 'g_Ca': 6.0, 'serotonin_factor': 1.5},
        network_mods={'excitatory_weight': 0.8, 'coupling_strength': 0.5},
        icc_mods={'omega': 0.000408, 'amplitude': 12.0},  # +30% → ~3.9 cpm
        muscle_mods={'tau_activation': 60.0, 'tau_relaxation': 150.0},
    ),

    'ibs_c': IBSProfile(
        name='IBS-C (Constipation-predominant)',
        membrane_mods={'g_Na': 80.0, 'g_K': 50.0, 'no_factor': 1.5},
        network_mods={'inhibitory_weight': 0.6, 'coupling_strength': 0.15},
        icc_mods={'omega': 0.000235, 'amplitude': 4.0},  # -25% → ~2.2 cpm
        muscle_mods={'tau_activation': 200.0, 'resting_tone': 0.1},
    ),

    'ibs_m': IBSProfile(
        name='IBS-M (Mixed)',
        membrane_mods={'g_Ca': 8.0, 'g_KCa': 3.0},
        network_mods={'coupling_strength': 0.1, 'reflex_gain': 0.2},
        icc_mods={'omega': 0.000377, 'coupling_icc': 0.03},  # Variable → ~3.6 cpm
        muscle_mods={'k_activation': 0.02},
    ),
}


class ENSGIDigitalTwin:
    """Unified ENS-GI Digital Twin — One Engine, Three Applications.
    
    Integrates all three layers into a single simulation system:
    - Layer 1: Cellular electrophysiology (ENSNeuron)
    - Layer 2: Network dynamics (ENSNetwork) 
    - Layer 3: ICC pacemaker + smooth muscle (ICCPacemaker + SmoothMuscle)
    
    Supports three operational modes:
    1. Research: Full parameter exploration, detailed recording
    2. Neuromorphic: SPICE-compatible output, hardware mapping
    3. Clinical: Patient-parameterized, biomarker extraction
    
    Usage:
        twin = ENSGIDigitalTwin(n_segments=20)
        twin.apply_profile('ibs_d')
        
        for _ in range(10000):
            result = twin.step(dt=0.05)
        
        biomarkers = twin.extract_biomarkers()
        spice_netlist = twin.export_spice_netlist()
    """
    
    def __init__(self, n_segments: int = 20,
                 membrane_params: Optional[MembraneParams] = None,
                 network_params: Optional[NetworkParams] = None,
                 icc_params: Optional[ICCParams] = None,
                 muscle_params: Optional[SmoothMuscleParams] = None):
        
        self.n_segments = n_segments
        
        # Initialize parameters
        mem_p = membrane_params or MembraneParams()
        net_p = network_params or NetworkParams(n_neurons=n_segments)
        net_p.n_neurons = n_segments
        icc_p = icc_params or ICCParams()
        mus_p = muscle_params or SmoothMuscleParams()
        
        # Build layers
        self.network = ENSNetwork(net_p, mem_p)
        self.icc = ICCPacemaker(n_segments, icc_p)
        self.muscle = SmoothMuscle(n_segments, mus_p)
        
        # Recording buffers
        self.recording = {
            'time': [],
            'voltages': [],
            'calcium': [],
            'icc_current': [],
            'force': [],
            'spikes': [],
        }
        
        self.time = 0.0
        self._profile = 'healthy'
    
    def apply_profile(self, profile_name: str):
        """Apply an IBS patient profile."""
        if profile_name not in IBS_PROFILES:
            raise ValueError(f"Unknown profile: {profile_name}. "
                           f"Available: {list(IBS_PROFILES.keys())}")
        
        profile = IBS_PROFILES[profile_name]
        self._profile = profile_name
        
        # Apply membrane modifications
        for neuron in self.network.neurons:
            for key, val in profile.membrane_mods.items():
                setattr(neuron.params, key, val)
        
        # Apply network modifications
        for key, val in profile.network_mods.items():
            setattr(self.network.params, key, val)
        
        # Apply ICC modifications
        for key, val in profile.icc_mods.items():
            setattr(self.icc.params, key, val)
        
        # Apply muscle modifications
        for key, val in profile.muscle_mods.items():
            setattr(self.muscle.params, key, val)
        
        # Rebuild connectivity with new params
        self.network.connections = self.network._build_connections()
    
    def step(self, dt: float = 0.05,
             I_stim: Optional[Dict[int, float]] = None,
             record: bool = True) -> Dict:
        """Advance the full digital twin by dt milliseconds.
        
        Execution order:
        1. ICC generates slow wave currents
        2. ENS network integrates (with ICC input)
        3. Smooth muscle computes force (from Ca²⁺ + neural drive)
        
        Returns: Complete state snapshot
        """
        # Layer 3a: ICC pacemaker
        I_icc = self.icc.step(dt)
        
        # Layer 1+2: ENS network
        net_result = self.network.step(dt, I_stim, I_icc)
        
        # Layer 3b: Smooth muscle contraction
        Ca_i = net_result['calcium']
        neural_drive = net_result['voltages'] / 100.0  # Normalize
        force = self.muscle.step(dt, Ca_i, neural_drive)
        
        self.time += dt
        
        result = {
            'time': self.time,
            'voltages': net_result['voltages'],
            'calcium': Ca_i,
            'icc_current': I_icc,
            'force': force.copy(),
            'spikes': net_result['spikes'],
            'icc_phase': self.icc.get_phase(),
        }
        
        if record:
            self.recording['time'].append(self.time)
            self.recording['voltages'].append(net_result['voltages'].copy())
            self.recording['calcium'].append(Ca_i.copy())
            self.recording['icc_current'].append(I_icc.copy())
            self.recording['force'].append(force.copy())
            self.recording['spikes'].append(net_result['spikes'])
        
        return result
    
    def run(self, duration: float, dt: float = 0.05,
            I_stim: Optional[Dict[int, float]] = None,
            record: bool = True,
            verbose: bool = False) -> Dict:
        """Run simulation for specified duration.
        
        Args:
            duration: Total simulation time (ms)
            dt: Time step (ms)
            I_stim: Static stimulation pattern
            record: Whether to record full history
            verbose: Print progress
            
        Returns: Final recording dict with numpy arrays
        """
        n_steps = int(duration / dt)
        
        for i in range(n_steps):
            self.step(dt, I_stim, record)
            if verbose and i % (n_steps // 10) == 0:
                print(f"  t = {self.time:.1f} ms ({100*i/n_steps:.0f}%)")
        
        # Convert to numpy arrays
        if record:
            return {
                'time': np.array(self.recording['time']),
                'voltages': np.array(self.recording['voltages']),
                'calcium': np.array(self.recording['calcium']),
                'icc_current': np.array(self.recording['icc_current']),
                'force': np.array(self.recording['force']),
            }
        return {}
    
    # ── Application 1: Research Simulator ──
    
    def parameter_sweep(self, param_name: str, values: np.ndarray,
                         duration: float = 500, dt: float = 0.05) -> List[Dict]:
        """Sweep a parameter and record steady-state behavior.
        
        Useful for bifurcation analysis, sensitivity studies.
        """
        results = []
        for val in values:
            # Reset
            self.__init__(self.n_segments)
            self.apply_profile(self._profile)
            
            # Set parameter
            for neuron in self.network.neurons:
                if hasattr(neuron.params, param_name):
                    setattr(neuron.params, param_name, val)
            if hasattr(self.icc.params, param_name):
                setattr(self.icc.params, param_name, val)
            
            # Run to steady state
            self.run(duration, dt, record=False)
            
            # Record snapshot
            state = self.step(dt)
            state['param_value'] = val
            results.append(state)
        
        return results
    
    # ── Application 2: Neuromorphic / Hardware Export ──
    
    def export_spice_netlist(self, filename: Optional[str] = None, use_verilog_a: bool = False) -> str:
        """Generate SPICE-compatible netlist for the ENS network.

        Maps biological components to circuit elements:
        - Membrane capacitance → C
        - Ion channels → Subcircuits (SPICE) or Verilog-A modules
        - Gap junctions → Resistors
        - ICC → Current source (simplified)

        Args:
            filename: Output filename (optional)
            use_verilog_a: If True, use Verilog-A modules; else pure SPICE subcircuits

        Returns:
            Complete SPICE netlist as string
        """
        lines = [
            "* ENS-GI Digital Twin — SPICE Netlist",
            f"* Profile: {self._profile}",
            f"* Segments: {self.n_segments}",
            "* Generated by ENSGIDigitalTwin",
            "*",
            "* USAGE:",
            "*   ngspice: ngspice netlist.sp",
            "*   HSPICE:  hspice -i netlist.sp",
            "*   Spectre: spectre netlist.sp",
            "",
        ]

        # Include Verilog-A modules if requested
        if use_verilog_a:
            lines.extend([
                "* --- Include Verilog-A Modules ---",
                ".hdl 'verilog_a_library/NaV1_5.va'",
                ".hdl 'verilog_a_library/Kv_delayed_rectifier.va'",
                ".hdl 'verilog_a_library/CaL_channel.va'",
                ".hdl 'verilog_a_library/leak_channel.va'",
                ".hdl 'verilog_a_library/gap_junction.va'",
                ".hdl 'verilog_a_library/icc_fhn_oscillator.va'",
                "",
            ])

        # Global parameters
        p0 = self.network.neurons[0].params
        lines.extend([
            "* --- Global Parameters ---",
            f".param C_m = {p0.C_m}e-12",
            f".param g_Na = {p0.g_Na}e-3",
            f".param g_K = {p0.g_K}e-3",
            f".param g_Ca = {p0.g_Ca}e-3",
            f".param g_L = {p0.g_L}e-3",
            f".param g_gap = {self.network.params.coupling_strength}e-3",
            f".param E_Na = {p0.E_Na}e-3",
            f".param E_K = {p0.E_K}e-3",
            f".param E_Ca = {p0.E_Ca}e-3",
            f".param E_L = {p0.E_L}e-3",
            "",
        ])

        # Subcircuit definitions (if not using Verilog-A)
        if not use_verilog_a:
            lines.extend(self._generate_spice_subcircuits())

        # Network instantiation
        lines.extend([
            "* ═══════════════════════════════════════════════════",
            "* ENS NETWORK INSTANTIATION",
            "* ═══════════════════════════════════════════════════",
            "",
        ])

        for i in range(self.n_segments):
            p = self.network.neurons[i].params

            lines.extend([
                f"* --- Neuron {i} ---",
                f"C_m{i} V{i} 0 {{C_m}}",
                "",
            ])

            if use_verilog_a:
                # Use Verilog-A modules
                lines.extend([
                    f"X_na{i} V{i} 0 NaV1_5 g_Na={{g_Na}} E_Na={{E_Na}}",
                    f"X_k{i}  V{i} 0 Kv_delayed_rectifier g_K={{g_K}} E_K={{E_K}}",
                    f"X_ca{i} V{i} 0 CaL_channel g_Ca={{g_Ca}} E_Ca={{E_Ca}}",
                    f"X_l{i}  V{i} 0 leak_channel g_L={{g_L}} E_L={{E_L}}",
                ])
            else:
                # Use SPICE subcircuits
                lines.extend([
                    f"X_na{i} V{i} 0 na_channel",
                    f"X_k{i}  V{i} 0 k_channel",
                    f"X_ca{i} V{i} 0 ca_channel",
                    f"X_l{i}  V{i} 0 leak",
                ])

            # ICC slow wave (simplified as current source)
            icc_amplitude = self.icc.params.amplitude
            icc_freq_hz = self.icc.params.omega / (2 * np.pi) * 1000  # rad/ms to Hz
            phase = i * self.icc.params.phase_gradient
            lines.append(f"I_icc{i} V{i} 0 SIN(0 {icc_amplitude}u {icc_freq_hz} 0 0 {phase})")

            # Gap junctions to neighbors
            if i < self.n_segments - 1:
                if use_verilog_a:
                    lines.append(f"X_gap{i} V{i} V{i+1} 0 gap_junction g_gap={{g_gap}}")
                else:
                    lines.append(f"R_gap{i} V{i} V{i+1} {{1/(g_gap+1n)}}")  # Add 1n to avoid /0

            lines.append("")

        # Stimulus (inject current into middle neuron)
        stim_neuron = self.n_segments // 2
        lines.extend([
            "* --- Stimulus ---",
            f"I_stim V{stim_neuron} 0 PULSE(0 10u 10m 1m 1m 50m 200m)",
            "",
        ])

        # Simulation control
        lines.extend([
            "* ═══════════════════════════════════════════════════",
            "* SIMULATION CONTROL",
            "* ═══════════════════════════════════════════════════",
            "",
            "* Transient analysis: 0.05ms steps, 1000ms duration",
            ".tran 0.05m 1000m",
            "",
            "* Output probes",
            ".print tran v(V0) v(V1) v(V2)",
            "",
            "* Convergence options",
            ".options reltol=1e-4 abstol=1e-9 vntol=1e-6",
            ".options method=gear",
            "",
            ".end",
        ])

        netlist = "\n".join(lines)

        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(netlist)
            print(f"[SPICE] Netlist written to {filename}")

        return netlist

    def _generate_spice_subcircuits(self) -> List[str]:
        """Generate SPICE-native subcircuit definitions (no Verilog-A).

        These are simplified behavioral models for compatibility with
        basic SPICE simulators that don't support Verilog-A.
        """
        return [
            "* ═══════════════════════════════════════════════════",
            "* SUBCIRCUIT DEFINITIONS (Behavioral Models)",
            "* ═══════════════════════════════════════════════════",
            "",
            "* Sodium channel (simplified - voltage-controlled current source)",
            ".subckt na_channel vp vn",
            "  G_na vp vn VALUE={(g_Na * pow((1/(1+exp(-(v(vp,vn)*1000+40)/10))), 3) * 0.6) * (v(vp,vn) - E_Na)}",
            ".ends",
            "",
            "* Potassium channel (simplified - voltage-controlled current source)",
            ".subckt k_channel vp vn",
            "  G_k vp vn VALUE={(g_K * pow((1/(1+exp(-(v(vp,vn)*1000+55)/10))), 4)) * (v(vp,vn) - E_K)}",
            ".ends",
            "",
            "* Calcium channel (L-type - voltage-controlled current source)",
            ".subckt ca_channel vp vn",
            "  .param g_Ca_local=2.0e-3 E_Ca_local=120e-3",
            "  .param V_half=-20.0 k=9.0",
            "  G_ca vp vn VALUE={(g_Ca_local * (1/(1+exp(-(v(vp,vn)*1000+V_half)/k)))) * (v(vp,vn) - E_Ca_local)}",
            ".ends",
            "",
            "* Leak channel (simple resistor)",
            ".subckt leak vp vn",
            "  G_l vp vn VALUE={g_L * (v(vp,vn) - E_L)}",
            ".ends",
            "",
            "* KCa channel (Ca-activated K+ - simplified without Ca dynamics)",
            "* Note: Full implementation requires Ca concentration tracking",
            ".subckt kca_channel vp vn",
            "  .param g_KCa_local=1.5e-3",
            "  .param Ca_half=0.5e-6 K_Ca=0.3e-6",
            "  * Simplified: assumes Ca ~ 0.5 μM (resting)",
            "  G_kca vp vn VALUE={(g_KCa_local * 0.3) * (v(vp,vn) - E_K)}",
            ".ends",
            "",
            "* A-type K channel (transient outward K+ current)",
            ".subckt a_type_k vp vn",
            "  .param g_A_local=1.0e-3",
            "  .param V_half_act=-30.0 k_act=15.0",
            "  .param V_half_inact=-60.0 k_inact=8.0",
            "  * Activation * Inactivation",
            "  G_a vp vn VALUE={(g_A_local * (1/(1+exp(-(v(vp,vn)*1000+V_half_act)/k_act))) * (1/(1+exp((v(vp,vn)*1000+V_half_inact)/k_inact)))) * (v(vp,vn) - E_K)}",
            ".ends",
            "",
        ]
    
    def export_verilog_a_module(self) -> str:
        """Generate Verilog-A module for ENS neuron.
        
        This is a behavioral model suitable for:
        - Cadence Spectre
        - Keysight ADS
        - LTspice (with Verilog-A support)
        """
        p = self.network.neurons[0].params
        
        return f"""
// ENS Neuron — Verilog-A Behavioral Model
// Generated by ENSGIDigitalTwin
// Profile: {self._profile}

`include "constants.vams"
`include "disciplines.vams"

module ens_neuron(V_mem, gnd);
    inout V_mem, gnd;
    electrical V_mem, gnd;
    
    // Parameters
    parameter real C_m = {p.C_m}e-6;      // F/cm²
    parameter real g_Na = {p.g_Na}e-3;     // S/cm²
    parameter real g_K = {p.g_K}e-3;
    parameter real g_Ca = {p.g_Ca}e-3;
    parameter real g_L = {p.g_L}e-3;
    parameter real E_Na = {p.E_Na}e-3;     // V
    parameter real E_K = {p.E_K}e-3;
    parameter real E_Ca = {p.E_Ca}e-3;
    parameter real E_L = {p.E_L}e-3;
    
    // State variables
    real V, m, h, n, m_Ca;
    real alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n;
    real I_Na, I_K, I_Ca, I_L;
    
    analog begin
        V = V(V_mem, gnd);
        
        // Na+ gating kinetics
        alpha_m = 0.1 * (V*1e3 + 40) / (1 - exp(-(V*1e3 + 40) / 10));
        beta_m = 4.0 * exp(-(V*1e3 + 65) / 18);
        alpha_h = 0.07 * exp(-(V*1e3 + 65) / 20);
        beta_h = 1.0 / (1 + exp(-(V*1e3 + 35) / 10));
        alpha_n = 0.01 * (V*1e3 + 55) / (1 - exp(-(V*1e3 + 55) / 10));
        beta_n = 0.125 * exp(-(V*1e3 + 65) / 80);
        
        // Gate dynamics
        m = alpha_m / (alpha_m + beta_m);  // Quasi-static for fast gate
        ddt(h) = alpha_h * (1 - h) - beta_h * h;
        ddt(n) = alpha_n * (1 - n) - beta_n * n;
        m_Ca = 1.0 / (1 + exp(-(V*1e3 + 20) / 9));
        
        // Ionic currents
        I_Na = g_Na * pow(m, 3) * h * (V - E_Na);
        I_K = g_K * pow(n, 4) * (V - E_K);
        I_Ca = g_Ca * m_Ca * (V - E_Ca);
        I_L = g_L * (V - E_L);
        
        // KCL at membrane node
        I(V_mem, gnd) <+ C_m * ddt(V);
        I(V_mem, gnd) <+ I_Na + I_K + I_Ca + I_L;
    end
endmodule
"""
    
    # ── Application 3: Clinical Biomarkers ──
    
    def extract_biomarkers(self) -> Dict:
        """Extract clinically relevant biomarkers from simulation.
        
        These biomarkers can be:
        - Compared to patient data
        - Used for IBS subtype classification
        - Fed into ML models for prediction
        """
        if not self.recording['time']:
            return {}
        
        voltages = np.array(self.recording['voltages'])
        forces = np.array(self.recording['force'])
        calcium = np.array(self.recording['calcium'])
        
        # Temporal analysis (use last 50% for steady state)
        n_half = len(voltages) // 2
        ss_voltages = voltages[n_half:]
        ss_forces = forces[n_half:]
        ss_calcium = calcium[n_half:]
        
        # Spike statistics
        all_spikes = []
        for spike_list in self.recording['spikes'][n_half:]:
            all_spikes.extend(spike_list)
        
        # Motility metrics
        mean_force = np.mean(ss_forces)
        force_variability = np.std(ss_forces)
        max_force = np.max(ss_forces)
        
        # Propagation analysis
        v_correlation = np.corrcoef(ss_voltages[:, 0], ss_voltages[:, -1])[0, 1] \
            if ss_voltages.shape[1] > 1 else 0
        
        biomarkers = {
            # Electrophysiology
            'mean_membrane_potential': float(np.mean(ss_voltages)),
            'voltage_variance': float(np.var(ss_voltages)),
            'spike_rate_per_neuron': len(all_spikes) / max(self.n_segments, 1) / max(len(ss_voltages) * 0.05 / 1000, 1e-6),
            
            # Calcium
            'mean_calcium': float(np.mean(ss_calcium)),
            'peak_calcium': float(np.max(ss_calcium)),
            
            # Motility
            'mean_contractile_force': float(mean_force),
            'force_variability': float(force_variability),
            'peak_force': float(max_force),
            'motility_index': float(mean_force * 100),
            
            # Network
            'propagation_correlation': float(v_correlation) if not np.isnan(v_correlation) else 0.0,
            
            # ICC
            'icc_frequency_cpm': self.icc.get_frequency(),
            
            # Profile
            'profile': self._profile,
        }
        
        return biomarkers
    
    def clinical_report(self) -> str:
        """Generate human-readable clinical interpretation."""
        bio = self.extract_biomarkers()
        if not bio:
            return "No simulation data available. Run simulation first."
        
        lines = [
            "=" * 60,
            "ENS-GI DIGITAL TWIN — CLINICAL REPORT",
            "=" * 60,
            f"Profile: {bio['profile']}",
            f"ICC Frequency: {bio['icc_frequency_cpm']:.1f} cpm "
            f"({'normal' if 2 < bio['icc_frequency_cpm'] < 4 else 'ABNORMAL'})",
            "",
            "ELECTROPHYSIOLOGY:",
            f"  Mean Vm: {bio['mean_membrane_potential']:.1f} mV",
            f"  Spike Rate: {bio['spike_rate_per_neuron']:.1f} Hz/neuron",
            f"  Voltage Variance: {bio['voltage_variance']:.1f} mV²",
            "",
            "MOTILITY:",
            f"  Motility Index: {bio['motility_index']:.1f}%",
            f"  Force Variability: {bio['force_variability']:.4f}",
            f"  Peak Force: {bio['peak_force']:.3f}",
            "",
            "CALCIUM:",
            f"  Mean [Ca²⁺]ᵢ: {bio['mean_calcium']*1000:.3f} μM",
            f"  Peak [Ca²⁺]ᵢ: {bio['peak_calcium']*1000:.3f} μM",
            "",
            "NETWORK:",
            f"  Propagation Correlation: {bio['propagation_correlation']:.3f}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Demo / Quick Start
# ═══════════════════════════════════════════════════════════════

def demo():
    """Quick demonstration of the ENS-GI Digital Twin."""
    print("╔══════════════════════════════════════════╗")
    print("║  ENS-GI Digital Twin — Phase 1 Demo      ║")
    print("╚══════════════════════════════════════════╝")
    print()
    
    # Create twin
    twin = ENSGIDigitalTwin(n_segments=12)
    
    # Run healthy baseline with stimulation to drive activity
    print("[1/4] Running healthy baseline (2000ms)...")
    twin.run(2000, dt=0.05, I_stim={3: 12.0, 4: 10.0}, verbose=True)
    healthy_bio = twin.extract_biomarkers()
    print(twin.clinical_report())
    
    # Run IBS-D
    print("\n[2/4] Running IBS-D profile (2000ms)...")
    twin_d = ENSGIDigitalTwin(n_segments=12)
    twin_d.apply_profile('ibs_d')
    twin_d.run(2000, dt=0.05, I_stim={3: 12.0, 4: 10.0})
    print(twin_d.clinical_report())
    
    # Run IBS-C
    print("[3/4] Running IBS-C profile (2000ms)...")
    twin_c = ENSGIDigitalTwin(n_segments=12)
    twin_c.apply_profile('ibs_c')
    twin_c.run(2000, dt=0.05, I_stim={3: 12.0, 4: 10.0})
    print(twin_c.clinical_report())
    
    # Export hardware models
    print("[4/4] Exporting hardware models...")
    netlist = twin.export_spice_netlist()
    print(f"  SPICE netlist: {len(netlist)} chars")
    
    va_module = twin.export_verilog_a_module()
    print(f"  Verilog-A module: {len(va_module)} chars")
    
    print("\n✓ Demo complete.")
    print("\nBiomarker comparison:")
    print(f"  {'Metric':<30} {'Healthy':>10} {'IBS-D':>10} {'IBS-C':>10}")
    print(f"  {'-'*60}")
    
    for key in ['motility_index', 'spike_rate_per_neuron', 'icc_frequency_cpm', 'mean_calcium']:
        h_val = healthy_bio.get(key, 0)
        d_val = twin_d.extract_biomarkers().get(key, 0)
        c_val = twin_c.extract_biomarkers().get(key, 0)
        print(f"  {key:<30} {h_val:>10.2f} {d_val:>10.2f} {c_val:>10.2f}")


if __name__ == '__main__':
    demo()
