"""
gpu_sim.py — GPU-batched ODE simulation for ENS-GI Digital Twin.

Replaces the CPU ProcessPoolExecutor with a single PyTorch CUDA pass that
evaluates all B samples simultaneously on the GPU.

State vector layout  [B, 13*N]:
  Channel   Slice         Description
  --------  ------------  --------------------------------------------------
  V         [ 0*N: 1*N]   Membrane potential (mV)
  m         [ 1*N: 2*N]   Na⁺ activation gate
  h         [ 2*N: 3*N]   Na⁺ inactivation gate
  n         [ 3*N: 4*N]   K⁺ activation gate
  m_Ca      [ 4*N: 5*N]   L-type Ca²⁺ activation gate
  a         [ 5*N: 6*N]   A-type K⁺ activation gate
  b         [ 6*N: 7*N]   A-type K⁺ inactivation gate
  s_e       [ 7*N: 8*N]   Excitatory synaptic variable
  s_i       [ 8*N: 9*N]   Inhibitory synaptic variable
  Ca_i      [ 9*N:10*N]   Intracellular Ca²⁺ concentration (mM)
  v_icc     [10*N:11*N]   ICC FitzHugh-Nagumo fast variable
  w_icc     [11*N:12*N]   ICC FitzHugh-Nagumo slow variable
  act       [12*N:13*N]   Smooth muscle activation (0–1)

Notes:
  - Synaptic spike-triggered increments are omitted (pure exponential decay).
    This is a valid simplification for synthetic training-data generation.
  - `omega` (ICC angular frequency) is stored in params_t but does NOT enter
    the FHN derivative equations — matching the current CPU behaviour where
    `setattr(twin.icc.params, 'omega', value)` has no effect on step().
  - I_stim is hardcoded: only neuron #3 receives 10 μA/cm² (same as the
    CPU worker _run_single_simulation).
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import torch


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Laplacian for gap-junction diffusion
# ═══════════════════════════════════════════════════════════════════════════════

def _build_laplacian(N: int, device: torch.device) -> torch.Tensor:
    """Build the nearest-neighbour graph Laplacian for N neurons.

    The gap-junction coupling current into neuron i is:
        I_couple[i] = g_gap * (L @ V)[i]

    Interior nodes (1 ≤ i ≤ N-2) have diagonal -2, boundary nodes -1.

    Returns
    -------
    L : torch.Tensor  shape [N, N], float32 on *device*
    """
    rows = []
    for i in range(N):
        row = torch.zeros(N, dtype=torch.float32)
        if i > 0:
            row[i - 1] += 1.0
            row[i]     -= 1.0
        if i < N - 1:
            row[i + 1] += 1.0
            row[i]     -= 1.0
        rows.append(row)
    return torch.stack(rows, dim=0).to(device)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Initial state
# ═══════════════════════════════════════════════════════════════════════════════

def _build_initial_state(
    B: int,
    N: int,
    icc_params,        # ICCParams dataclass instance
    muscle_params,     # SmoothMuscleParams dataclass instance
    device: torch.device,
) -> torch.Tensor:
    """Build the initial state tensor of shape [B, 13*N].

    Neuron ICs are set to the HH steady-state at V = −65 mV with ±3 mV
    uniform jitter per sample (matching ENSNeuron.__init__ behaviour).
    ICC ICs use the phase-gradient initialisation from ICCPacemaker.__init__.
    Muscle IC: resting_tone on every segment.

    Returns
    -------
    state : torch.Tensor  float32 on *device*
    """
    state = torch.zeros(B, 13 * N, device=device, dtype=torch.float32)

    # ── Membrane potential: −65 ± 3 mV ──────────────────────────────────────
    V_jitter = torch.rand(B, N, device=device) * 6.0 - 3.0
    state[:, 0 * N:1 * N] = -65.0 + V_jitter

    # ── HH gating variables at steady state of V = −65 mV ───────────────────
    # Computed analytically:
    #   alpha_m(-65) ≈ 0.2236   beta_m(-65) = 4.0   → m_ss ≈ 0.053
    #   alpha_h(-65) = 0.07     beta_h(-65) ≈ 0.047 → h_ss ≈ 0.596
    #   alpha_n(-65) ≈ 0.0695   beta_n(-65) = 0.125 → n_ss ≈ 0.357
    state[:, 1 * N:2 * N] = 0.053   # m  (Na activation)
    state[:, 2 * N:3 * N] = 0.596   # h  (Na inactivation)
    state[:, 3 * N:4 * N] = 0.357   # n  (K activation)
    state[:, 4 * N:5 * N] = 0.0     # m_Ca
    state[:, 5 * N:6 * N] = 0.0     # a  (A-type activation)
    state[:, 6 * N:7 * N] = 1.0     # b  (A-type inactivation, starts closed→1)

    # ── Synaptic & calcium ───────────────────────────────────────────────────
    # s_e, s_i = 0;  Ca_i = 0.0001 mM (default resting)
    state[:, 7 * N:8 * N]  = 0.0
    state[:, 8 * N:9 * N]  = 0.0
    state[:, 9 * N:10 * N] = 1e-4

    # ── ICC FHN: phase-gradient initialisation ────────────────────────────────
    phase_gradient = float(getattr(icc_params, 'phase_gradient', 0.3))
    seg_idx = torch.arange(N, dtype=torch.float32, device=device)
    phases  = seg_idx * phase_gradient
    v0 = torch.cos(phases) * 0.5   # shape [N]
    w0 = torch.sin(phases) * 0.3
    # Broadcast identical IC across all B samples
    state[:, 10 * N:11 * N] = v0.unsqueeze(0).expand(B, N)
    state[:, 11 * N:12 * N] = w0.unsqueeze(0).expand(B, N)

    # ── Smooth muscle: resting activation ────────────────────────────────────
    resting_tone = float(getattr(muscle_params, 'resting_tone', 0.05))
    state[:, 12 * N:13 * N] = resting_tone

    return state


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  ODE derivative function
# ═══════════════════════════════════════════════════════════════════════════════

def _ens_derivatives(
    state: torch.Tensor,         # [B, 13*N]
    params_t: torch.Tensor,      # [B, 5]  [g_Na, g_K, g_Ca, omega, coupling_s]
    lap: torch.Tensor,           # [N, N]
    bp: Dict[str, torch.Tensor], # scalar tensors of base biophysical params
    I_stim_vec: torch.Tensor,    # [N]  external current (fixed per run)
    N: int,
) -> torch.Tensor:               # [B, 13*N]
    """Compute d/dt of the full [B, 13*N] state vector.

    All operations are fully vectorised over the batch dimension B and the
    spatial dimension N simultaneously — no Python loops at all.

    HH α/β singularities (V = −40, −55) are regularised with + 1e-7 in the
    denominator, matching the convention used in pinn.py's TF functions.

    Parameter column mapping in params_t:
        col 0 → g_Na  (per-sample, replaces base g_Na for all N neurons)
        col 1 → g_K
        col 2 → g_Ca
        col 3 → omega (stored but unused in dynamics — matches CPU behaviour)
        col 4 → coupling_strength (gap-junction conductance, g_gap)
    """
    # ── Unpack state slices → each [B, N] ───────────────────────────────────
    V     = state[:, 0 * N:1 * N]
    m     = state[:, 1 * N:2 * N]
    h     = state[:, 2 * N:3 * N]
    n     = state[:, 3 * N:4 * N]
    m_Ca  = state[:, 4 * N:5 * N]
    a     = state[:, 5 * N:6 * N]
    b_var = state[:, 6 * N:7 * N]
    s_e   = state[:, 7 * N:8 * N]
    s_i   = state[:, 8 * N:9 * N]
    Ca_i  = state[:, 9 * N:10 * N]
    v_icc = state[:, 10 * N:11 * N]
    w_icc = state[:, 11 * N:12 * N]
    act   = state[:, 12 * N:13 * N]

    # ── Per-sample conductances: [B, 1] for broadcasting to [B, N] ──────────
    g_Na_s = params_t[:, 0:1]          # [B,1]
    g_K_s  = params_t[:, 1:2]
    g_Ca_s = params_t[:, 2:3]
    # params_t[:, 3] = omega (ignored in dynamics)
    coup_s = params_t[:, 4:5]          # [B,1] coupling_strength = g_gap

    # ═══════════════════════════════════════════════════════
    # (a)  HH α/β gating rates
    # ═══════════════════════════════════════════════════════

    # Na activation  (singularity at V = −40)
    dV_m    = V + 40.0
    alpha_m = 0.1 * dV_m / (1.0 - torch.exp(-dV_m / 10.0) + 1e-7)
    beta_m  = 4.0 * torch.exp(-(V + 65.0) / 18.0)

    # Na inactivation
    alpha_h = 0.07 * torch.exp(-(V + 65.0) / 20.0)
    beta_h  = 1.0 / (1.0 + torch.exp(-(V + 35.0) / 10.0))

    # K activation  (singularity at V = −55)
    dV_n    = V + 55.0
    alpha_n = 0.01 * dV_n / (1.0 - torch.exp(-dV_n / 10.0) + 1e-7)
    beta_n  = 0.125 * torch.exp(-(V + 65.0) / 80.0)

    # ═══════════════════════════════════════════════════════
    # (b)  Ca²⁺ gating (L-type channel, enteric-specific)
    # ═══════════════════════════════════════════════════════
    m_Ca_inf  = 1.0 / (1.0 + torch.exp(-(V + 20.0) / 9.0))
    tau_m_Ca  = 1.0 + 10.0 / (1.0 + torch.exp((V + 20.0) / 9.0))

    # ═══════════════════════════════════════════════════════
    # (c)  A-type K⁺ gating (τ_a = 5 ms, τ_b = 50 ms)
    # ═══════════════════════════════════════════════════════
    a_inf = 1.0 / (1.0 + torch.exp(-(V + 45.0) / 14.5))
    b_inf = 1.0 / (1.0 + torch.exp((V + 70.0) / 7.5))

    # ═══════════════════════════════════════════════════════
    # (d)  Ion currents  I = g * (V − E)
    # ═══════════════════════════════════════════════════════
    I_Na  = g_Na_s * m ** 3 * h            * (V - bp['E_Na'])
    I_K   = g_K_s  * n ** 4                * (V - bp['E_K'])
    I_L   =          bp['g_L']             * (V - bp['E_L'])
    I_Ca  = g_Ca_s * m_Ca                  * (V - bp['E_Ca'])

    # Ca²⁺-activated K⁺ (AHP)
    KCa_act = Ca_i / (Ca_i + bp['Ca_half'])
    I_KCa   = bp['g_KCa'] * KCa_act * (V - bp['E_K'])

    # A-type K⁺
    I_A = bp['g_A'] * a * b_var * (V - bp['E_K'])

    # Synaptic
    I_syn_e = bp['g_syn_e'] * s_e * (V - bp['E_syn_e'])
    I_syn_i = bp['g_syn_i'] * s_i * (V - bp['E_syn_i'])

    # ═══════════════════════════════════════════════════════
    # (e)  Gap-junction coupling current
    #      I_couple[b, :] = coup_s[b] * (V[b, :] @ L)
    #      L = L^T (symmetric), so V @ L = V @ L^T
    #      Using @ (matmul) tolerates non-contiguous slices of state.
    # ═══════════════════════════════════════════════════════
    I_couple = coup_s * (V @ lap)   # [B, N]

    # ═══════════════════════════════════════════════════════
    # (f)  ICC pacemaker input to neurons
    #      I_icc[b, i] = amplitude * v_icc[b, i]
    # ═══════════════════════════════════════════════════════
    I_icc = bp['icc_amplitude'] * v_icc        # [B, N]

    # ═══════════════════════════════════════════════════════
    # (g)  External stimulation  [1, N] broadcast to [B, N]
    # ═══════════════════════════════════════════════════════
    I_ext = I_stim_vec.unsqueeze(0)            # [1, N]

    # ═══════════════════════════════════════════════════════
    # (h)  dV/dt
    # ═══════════════════════════════════════════════════════
    I_ion_total = I_Na + I_K + I_L + I_Ca + I_KCa + I_A + I_syn_e + I_syn_i
    dV = (-I_ion_total + I_ext + I_couple + I_icc) / bp['C_m']

    # ═══════════════════════════════════════════════════════
    # (i)  Gating-variable derivatives
    # ═══════════════════════════════════════════════════════
    dm    = alpha_m * (1.0 - m)   - beta_m * m
    dh    = alpha_h * (1.0 - h)   - beta_h * h
    dn    = alpha_n * (1.0 - n)   - beta_n * n
    dm_Ca = (m_Ca_inf - m_Ca) / tau_m_Ca
    da    = (a_inf   - a)     / 5.0    # tau_a = 5 ms
    db    = (b_inf   - b_var) / 50.0   # tau_b = 50 ms

    # ═══════════════════════════════════════════════════════
    # (j)  Synaptic decay  (spike-triggered increments omitted)
    # ═══════════════════════════════════════════════════════
    ds_e = -s_e / bp['tau_syn_e']
    ds_i = -s_i / bp['tau_syn_i']

    # ═══════════════════════════════════════════════════════
    # (k)  Intracellular Ca²⁺ dynamics
    # ═══════════════════════════════════════════════════════
    dCa = -bp['k_Ca'] * I_Ca - Ca_i / bp['tau_Ca']

    # ═══════════════════════════════════════════════════════
    # (l)  ICC FitzHugh-Nagumo dynamics + ICC-to-ICC coupling
    #      dv = v − v³/3 − w + I_coupling
    #      dw = ε(v + a_fhn − b_fhn·w)
    # ═══════════════════════════════════════════════════════
    icc_coup = torch.zeros_like(v_icc)
    icc_coup[:, 1:]  += bp['coupling_icc'] * (v_icc[:, :-1] - v_icc[:, 1:])
    icc_coup[:, :-1] += bp['coupling_icc'] * (v_icc[:, 1:]  - v_icc[:, :-1])

    dv_icc = v_icc - v_icc ** 3 / 3.0 - w_icc + icc_coup
    dw_icc = bp['epsilon'] * (v_icc + bp['a_fhn'] - bp['b_fhn'] * w_icc)

    # ═══════════════════════════════════════════════════════
    # (m)  Smooth muscle first-order activation
    #      Asymmetric τ: τ_act when target > act, τ_relax otherwise
    #      Hill function: Ca_activation = Ca^n / (Ca^n + Ca_half^n)
    #      neural_drive  = V / 100  (same normalisation as core.py)
    # ═══════════════════════════════════════════════════════
    Ca_n = Ca_i ** bp['hill_n']
    Ca_act_hill = Ca_n / (Ca_n + bp['Ca_half_force_n'])
    neural_drive = V / 100.0
    total_drive  = Ca_act_hill + bp['k_activation'] * neural_drive
    target_act   = torch.clamp(total_drive + bp['resting_tone'], 0.0, 1.0)

    tau_muscle = torch.where(
        target_act > act,
        bp['tau_activation'],
        bp['tau_relaxation'],
    )
    dact = (target_act - act) / tau_muscle

    # ═══════════════════════════════════════════════════════
    # (n)  Concatenate → [B, 13*N]
    # ═══════════════════════════════════════════════════════
    return torch.cat(
        [dV, dm, dh, dn, dm_Ca, da, db, ds_e, ds_i, dCa, dv_icc, dw_icc, dact],
        dim=1,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Fixed-step RK4 integrator + state clamping
# ═══════════════════════════════════════════════════════════════════════════════

def _rk4_step(
    state: torch.Tensor,
    dt: float,
    deriv_fn,
) -> torch.Tensor:
    """One classical RK4 step over the full [B, 13*N] state tensor.

    Intermediate k-tensors are computed without state clamping (matching the
    CPU implementation where clipping is applied only after the full step).
    """
    k1 = deriv_fn(state)
    k2 = deriv_fn(state + 0.5 * dt * k1)
    k3 = deriv_fn(state + 0.5 * dt * k2)
    k4 = deriv_fn(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _clamp_state(state: torch.Tensor, N: int) -> torch.Tensor:
    """Enforce physical bounds after each RK4 step (matches ENSNeuron.step).

    Gating variables m, h, n, m_Ca, a, b  → clamped to [0, 1]
    Synaptic variables s_e, s_i           → clamped to [0, ∞)
    Intracellular Ca²⁺                    → clamped to [0, ∞)
    Smooth muscle activation              → clamped to [0, ∞)
    V and ICC FHN variables unconstrained.

    Uses in-place clamp_ because _rk4_step always returns a freshly-allocated
    tensor — no aliasing risk and no extra allocation per step.
    """
    state[:, 1 * N:7 * N].clamp_(0.0, 1.0)   # m, h, n, m_Ca, a, b
    state[:, 7 * N:9 * N].clamp_(min=0.0)     # s_e, s_i
    state[:, 9 * N:10 * N].clamp_(min=0.0)    # Ca_i
    state[:, 12 * N:13 * N].clamp_(min=0.0)   # act
    return state


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Public entry point
# ═══════════════════════════════════════════════════════════════════════════════

def batch_simulate_gpu(
    all_params: np.ndarray,   # [B, 5]: g_Na, g_K, g_Ca, omega, coupling_strength
    base_twin_cfg: dict,       # same structure as pinn.py's twin_cfg dict
    n_segments: int,
    duration: float,           # ms
    dt: float,                 # ms
    noise_level: float,        # fraction of signal std
    record_steps: int = 500,   # target number of recorded time points
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate all B parameter sets simultaneously on the GPU.

    Parameters
    ----------
    all_params : ndarray [B, 5]
        Per-sample parameters: [g_Na, g_K, g_Ca, omega, coupling_strength].
    base_twin_cfg : dict
        Reference twin configuration as produced by pinn.py::generate_synthetic_dataset
        (keys: n_segments, neuron_params, network_params, icc_params, muscle_params).
    n_segments : int
        Number of gut segments N.
    duration : float
        Simulation duration in ms.
    dt : float
        Integration time step in ms.
    noise_level : float
        Gaussian noise added post-simulation (std = noise_level * signal_std).
    record_steps : int
        Approximate number of time points stored.  Actual stored count is
        ceil(n_steps / record_every).

    Returns
    -------
    voltages  : float32 ndarray  [B, T_stored, N]
    forces    : float32 ndarray  [B, T_stored, N]  (smooth-muscle activation * max_force)
    calcium   : float32 ndarray  [B, T_stored, N]
    """
    device = torch.device('cuda')

    B = all_params.shape[0]
    N = n_segments
    n_steps = int(duration / dt)

    # ── Base biophysical parameter dict ─────────────────────────────────────
    bp0 = base_twin_cfg['neuron_params'][0]    # MembraneParams (same for all neurons)
    icc = base_twin_cfg['icc_params']
    mus = base_twin_cfg['muscle_params']

    # Scalar tensors on GPU  (avoids re-cast inside the hot loop)
    def _t(v: float) -> torch.Tensor:
        return torch.tensor(float(v), device=device, dtype=torch.float32)

    bp: Dict[str, torch.Tensor] = {
        # Membrane
        'C_m':       _t(bp0.C_m),
        'g_L':       _t(bp0.g_L),
        'E_Na':      _t(bp0.E_Na),
        'E_K':       _t(bp0.E_K),
        'E_L':       _t(bp0.E_L),
        'E_Ca':      _t(bp0.E_Ca),
        'g_KCa':     _t(bp0.g_KCa),
        'g_A':       _t(bp0.g_A),
        'Ca_half':   _t(bp0.Ca_half),
        'g_syn_e':   _t(bp0.g_syn_e),
        'g_syn_i':   _t(bp0.g_syn_i),
        'E_syn_e':   _t(bp0.E_syn_e),
        'E_syn_i':   _t(bp0.E_syn_i),
        'tau_syn_e': _t(bp0.tau_syn_e),
        'tau_syn_i': _t(bp0.tau_syn_i),
        'tau_Ca':    _t(bp0.tau_Ca),
        'k_Ca':      _t(bp0.k_Ca),
        # ICC
        'icc_amplitude': _t(icc.amplitude),
        'epsilon':       _t(icc.epsilon),
        'a_fhn':         _t(icc.a_fhn),
        'b_fhn':         _t(icc.b_fhn),
        'coupling_icc':  _t(icc.coupling_icc),
        # Smooth muscle
        'hill_n':           _t(mus.hill_n),
        'Ca_half_force_n':  _t(mus.Ca_half_force ** mus.hill_n),  # precomputed
        'k_activation':     _t(mus.k_activation),
        'tau_activation':   _t(mus.tau_activation),
        'tau_relaxation':   _t(mus.tau_relaxation),
        'resting_tone':     _t(mus.resting_tone),
        'max_force':        _t(mus.max_force),
    }

    # ── Precomputed fixed tensors ────────────────────────────────────────────
    lap = _build_laplacian(N, device)                       # [N, N]

    I_stim_vec = torch.zeros(N, device=device, dtype=torch.float32)
    I_stim_vec[3] = 10.0   # neuron #3 receives 10 μA/cm²

    params_t = torch.tensor(all_params, device=device, dtype=torch.float32)  # [B, 5]

    # ── Initial state ────────────────────────────────────────────────────────
    state = _build_initial_state(B, N, icc, mus, device)   # [B, 13*N]

    # ── Recording setup ──────────────────────────────────────────────────────
    record_every = max(1, n_steps // record_steps)
    # Exact number of frames that will be stored
    T_stored = math.ceil(n_steps / record_every)

    V_rec   = torch.empty(B, T_stored, N, device=device, dtype=torch.float32)
    Ca_rec  = torch.empty(B, T_stored, N, device=device, dtype=torch.float32)
    Act_rec = torch.empty(B, T_stored, N, device=device, dtype=torch.float32)

    # ── Closure captures all precomputed tensors ─────────────────────────────
    def deriv(s: torch.Tensor) -> torch.Tensor:
        return _ens_derivatives(s, params_t, lap, bp, I_stim_vec, N)

    # ── Time-integration loop ────────────────────────────────────────────────
    rec_idx = 0
    with torch.no_grad():
        for step in range(n_steps):
            state = _rk4_step(state, dt, deriv)
            state = _clamp_state(state, N)

            if step % record_every == 0:
                V_rec[:, rec_idx, :]   = state[:, 0 * N:1 * N]
                Ca_rec[:, rec_idx, :]  = state[:, 9 * N:10 * N]
                Act_rec[:, rec_idx, :] = state[:, 12 * N:13 * N] * bp['max_force']
                rec_idx += 1

    # ── Add noise (per-sample, matching CPU worker) ──────────────────────────
    if noise_level > 0.0:
        _add_noise_inplace(V_rec,   noise_level, device)
        _add_noise_inplace(Ca_rec,  noise_level, device)
        _add_noise_inplace(Act_rec, noise_level, device)

    # ── Transfer to CPU numpy ────────────────────────────────────────────────
    voltages = V_rec.cpu().numpy()     # [B, T_stored, N]
    calcium  = Ca_rec.cpu().numpy()
    forces   = Act_rec.cpu().numpy()

    return voltages, forces, calcium


# ─────────────────────────────────────────────────────────────────────────────

def _add_noise_inplace(
    arr: torch.Tensor,
    noise_level: float,
    device: torch.device,
) -> None:
    """Add zero-mean Gaussian noise in-place (σ = noise_level * arr.std_per_sample)."""
    # Per-sample standard deviation: shape [B, 1, 1]
    std = arr.std(dim=(1, 2), keepdim=True).clamp(min=1e-8)
    noise = torch.randn_like(arr) * (noise_level * std)
    arr.add_(noise)
