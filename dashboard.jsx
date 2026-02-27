import { useState, useEffect, useRef, useCallback, useMemo } from "react";

// ─── Simulation Engine (JavaScript port of core ENS-GI model) ───
const PHYSIO_OMEGA_BASE = 0.000314; // rad/ms ~= 3 cpm
const ICC_PHASE_SPEEDUP = 120; // visual acceleration while keeping physiological omega values

const PARAMS = {
  // Membrane properties
  C_m: 1.0, // μF/cm²
  // Ion channel conductances (mS/cm²)
  g_Na: 120.0, g_K: 36.0, g_L: 0.3, g_Ca: 4.0,
  g_KCa: 5.0,       // Ca2+-activated K+ conductance (mS/cm²)
  g_A: 8.0,         // A-type K+ conductance (mS/cm²)
  // Reversal potentials (mV)
  E_Na: 50.0, E_K: -77.0, E_L: -54.4, E_Ca: 120.0,
  // KCa parameters
  Ca_half: 0.001,   // Half-activation concentration for KCa (mM)
  // Synaptic
  g_syn_e: 0.5, g_syn_i: 1.0, E_syn_e: 0.0, E_syn_i: -80.0,
  tau_syn: 5.0,
  // ICC pacemaker
  omega_icc: PHYSIO_OMEGA_BASE,
  A_icc: 5.0,
  // Smooth muscle
  k_ca: 0.0002, tau_ca: 50.0,  // Unified with Python: faster Ca2+ clearance
  Ca_half_force: 0.02,          // Ca2+ for half-maximal force (mM)
  hill_n: 3.0,                  // Hill coefficient (cooperativity)
  // Network coupling
  coupling_strength: 0.3,
};

const IBS_PARAMS = {
  hypersensitive: { g_Na: 150, g_Ca: 6.0, g_syn_e: 0.8, omega_icc: PHYSIO_OMEGA_BASE * 1.3 },
  hyposensitive: { g_Na: 80, g_K: 50, g_syn_i: 2.0, omega_icc: PHYSIO_OMEGA_BASE * 0.75 },
  dysrhythmic: { g_Ca: 8.0, omega_icc: PHYSIO_OMEGA_BASE * 1.15, coupling_strength: 0.1 },
};

function alphaM(V) { return 0.1 * (V + 40) / (1 - Math.exp(-(V + 40) / 10)); }
function betaM(V) { return 4.0 * Math.exp(-(V + 65) / 18); }
function alphaH(V) { return 0.07 * Math.exp(-(V + 65) / 20); }
function betaH(V) { return 1.0 / (1 + Math.exp(-(V + 35) / 10)); }
function alphaN(V) { return 0.01 * (V + 55) / (1 - Math.exp(-(V + 55) / 10)); }
function betaN(V) { return 0.125 * Math.exp(-(V + 65) / 80); }
function mCaInf(V) { return 1.0 / (1 + Math.exp(-(V + 20) / 9)); }
// KCa and A-type K+ gating functions
function KCa_inf(Ca, Ca_half) { return Ca / (Ca + Ca_half); }
function a_inf(V) { return 1.0 / (1.0 + Math.exp(-(V + 45.0) / 14.5)); }
function b_inf(V) { return 1.0 / (1.0 + Math.exp((V + 70.0) / 7.5)); }

class ENSNeuron {
  constructor(params, id = 0) {
    this.p = { ...PARAMS, ...params };
    this.id = id;
    this.V = -65 + Math.random() * 5;
    this.m = alphaM(this.V) / (alphaM(this.V) + betaM(this.V));
    this.h = alphaH(this.V) / (alphaH(this.V) + betaH(this.V));
    this.n = alphaN(this.V) / (alphaN(this.V) + betaN(this.V));
    this.a = a_inf(this.V);  // A-type K+ activation
    this.b = b_inf(this.V);  // A-type K+ inactivation
    this.s_e = 0; this.s_i = 0;
    this.ca = 0.0001; this.force = 0;
    this.spiked = false;
  }

  step(dt, I_stim = 0, I_couple = 0, icc_phase = 0) {
    const { V, m, h, n, a, b, s_e, s_i, ca } = this;
    const p = this.p;

    const I_Na = p.g_Na * m * m * m * h * (V - p.E_Na);
    const I_K = p.g_K * n * n * n * n * (V - p.E_K);
    const I_L = p.g_L * (V - p.E_L);
    const I_Ca = p.g_Ca * mCaInf(V) * (V - p.E_Ca);
    const I_KCa = p.g_KCa * KCa_inf(ca, p.Ca_half) * (V - p.E_K);  // Ca2+-activated K+
    const I_A = p.g_A * a * b * (V - p.E_K);  // A-type K+
    const I_syn = p.g_syn_e * s_e * (V - p.E_syn_e) + p.g_syn_i * s_i * (V - p.E_syn_i);
    const I_icc = p.A_icc * Math.sin(icc_phase);

    const dV = (-I_Na - I_K - I_L - I_Ca - I_KCa - I_A - I_syn + I_stim + I_couple + I_icc) / p.C_m;
    const dm = alphaM(V) * (1 - m) - betaM(V) * m;
    const dh = alphaH(V) * (1 - h) - betaH(V) * h;
    const dn = alphaN(V) * (1 - n) - betaN(V) * n;
    const tau_a = 5.0;   // ms
    const tau_b = 50.0;  // ms
    const da = (a_inf(V) - a) / tau_a;
    const db = (b_inf(V) - b) / tau_b;
    const ds_e = -s_e / p.tau_syn;
    const ds_i = -s_i / p.tau_syn;
    const dca = -p.k_ca * I_Ca - ca / p.tau_ca;
    
    this.V = V + dV * dt;
    this.m = Math.max(0, Math.min(1, m + dm * dt));
    this.h = Math.max(0, Math.min(1, h + dh * dt));
    this.n = Math.max(0, Math.min(1, n + dn * dt));
    this.a = Math.max(0, Math.min(1, a + da * dt));
    this.b = Math.max(0, Math.min(1, b + db * dt));
    this.s_e = Math.max(0, s_e + ds_e * dt);
    this.s_i = Math.max(0, s_i + ds_i * dt);
    this.ca = Math.max(0, ca + dca * dt);

    // Hai-Murphy Hill function for smooth muscle force
    const ca_n = Math.pow(this.ca, p.hill_n);
    const ca_half_n = Math.pow(p.Ca_half_force, p.hill_n);
    this.force = Math.max(0, Math.min(1, ca_n / (ca_n + ca_half_n)));
    
    this.spiked = this.V > 0 && V <= 0;
    if (this.V > 60) this.V = 60;
    if (this.V < -90) this.V = -90;
  }
}

class ENSNetwork {
  constructor(n = 12, params = {}) {
    this.neurons = Array.from({ length: n }, (_, i) => new ENSNeuron(params, i));
    this.n = n;
    this.time = 0;
    this.icc_phase = 0;
    this.params = { ...PARAMS, ...params };
  }

  step(dt = 0.05, stim_neuron = -1, stim_current = 10) {
    this.icc_phase += this.params.omega_icc * dt * ICC_PHASE_SPEEDUP;
    
    for (let i = 0; i < this.n; i++) {
      let I_couple = 0;
      if (i > 0) I_couple += this.params.coupling_strength * (this.neurons[i-1].V - this.neurons[i].V);
      if (i < this.n - 1) I_couple += this.params.coupling_strength * (this.neurons[i+1].V - this.neurons[i].V);
      
      const I_stim = (i === stim_neuron) ? stim_current : 0;
      const phase_offset = this.icc_phase - (i * 0.5);
      
      this.neurons[i].step(dt, I_stim, I_couple, phase_offset);
      
      if (this.neurons[i].spiked) {
        if (i > 0) this.neurons[i-1].s_e += 0.5;
        if (i < this.n - 1) this.neurons[i+1].s_e += 0.5;
        if (i > 1) this.neurons[i-2].s_i += 0.3;
        if (i < this.n - 2) this.neurons[i+2].s_i += 0.3;
      }
    }
    this.time += dt;
  }
}

// ─── Visualization Components ───

const COLORS = {
  bg: "#0a0e17",
  bgDeep: "#070b12",
  panel: "#111827",
  panelHover: "#162034",
  panelBorder: "#1e293b",
  panelL1: "#0e1f1a",
  panelL1Border: "#12352c",
  panelL2: "#0e1a22",
  panelL2Border: "#123146",
  panelL3: "#201610",
  panelL3Border: "#3a2516",
  accent1: "#00e5a0",
  accent2: "#00b4d8",
  accent3: "#e040fb",
  accent4: "#ff6b35",
  text: "#e2e8f0",
  textMicro: "#94a3b8",
  textDim: "#64748b",
  textMuted: "#334155",
  danger: "#ef4444",
  warning: "#f59e0b",
  runGlow: "rgba(0, 229, 160, 0.55)",
  divider: "#1b2434",
  gridLine: "#1a2332",
};

function MiniScope({ data, color, width = 240, height = 90, label, value, unit }) {
  const points = useMemo(() => {
    if (!data || data.length === 0) return "";
    const maxAbs = Math.max(...data.map(Math.abs), 1);
    return data.map((v, i) => {
      const x = (i / (data.length - 1)) * width;
      const y = height / 2 - (v / maxAbs) * (height / 2 - 4);
      return `${x},${y}`;
    }).join(" ");
  }, [data, width, height]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {(label || value !== undefined) && (
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
          <div style={{ fontSize: 11, color: COLORS.textDim, fontFamily: "'JetBrains Mono', monospace", letterSpacing: "0.06em", textTransform: "uppercase" }}>
            {label}
          </div>
          {value !== undefined && (
            <div style={{ fontSize: 12, color, fontFamily: "'JetBrains Mono', monospace", fontWeight: 600 }}>
              {typeof value === "number" ? value.toFixed(1) : value}{unit && <span style={{ fontSize: 11, opacity: 0.7 }}> {unit}</span>}
            </div>
          )}
        </div>
      )}
      <svg width={width} height={height} style={{ display: "block" }}>
        <rect width={width} height={height} fill={COLORS.bgDeep} rx={6} />
        {[0.2, 0.4, 0.6, 0.8].map(p => (
          <line key={p} x1={0} y1={height * p} x2={width} y2={height * p} stroke={COLORS.gridLine} strokeWidth={0.5} />
        ))}
        <line x1={0} y1={height/2} x2={width} y2={height/2} stroke={COLORS.textMuted} strokeWidth={0.6} strokeDasharray="3,3" />
        {points && <polyline points={points} fill="none" stroke={color} strokeWidth={2} opacity={0.9} />}
      </svg>
    </div>
  );
}

function NeuronGrid({ neurons, width = 400, height = 200 }) {
  const n = neurons.length;
  const cellW = width / n;
  
  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      <defs>
        <linearGradient id="spike-grad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={COLORS.accent1} stopOpacity={0.8} />
          <stop offset="100%" stopColor={COLORS.accent1} stopOpacity={0} />
        </linearGradient>
        <filter id="glow">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
      </defs>
      <rect width={width} height={height} fill={COLORS.bg} rx={6} />
      
      {neurons.map((neuron, i) => {
        const x = i * cellW + cellW / 2;
        const vNorm = (neuron.V + 90) / 150;
        const barH = vNorm * (height - 20);
        const isActive = neuron.V > -20;
        const forceH = neuron.force * 30;
        
        return (
          <g key={i}>
            <rect x={x - cellW/2 + 1} y={height - barH - 10} width={cellW - 2} height={barH} 
              fill={isActive ? COLORS.accent1 : COLORS.accent2} opacity={0.15 + vNorm * 0.6} rx={2} />
            {isActive && (
              <circle cx={x} cy={height - barH - 10} r={4} fill={COLORS.accent1} filter="url(#glow)" opacity={0.9}>
                <animate attributeName="r" values="3;6;3" dur="0.3s" repeatCount="1" />
              </circle>
            )}
            <rect x={x - cellW/4} y={height - forceH - 4} width={cellW/2} height={forceH} 
              fill={COLORS.accent4} opacity={0.5} rx={1} />
            <text x={x} y={height - 1} textAnchor="middle" fontSize={9} fill={COLORS.textDim} fontFamily="'JetBrains Mono', monospace">
              {i}
            </text>
          </g>
        );
      })}
      
      <text x={6} y={14} fontSize={11} fill={COLORS.textDim} fontFamily="'JetBrains Mono', monospace">
        NETWORK ACTIVITY MAP
      </text>
    </svg>
  );
}

function PropagationWave({ neurons, width = 400, height = 100 }) {
  const n = neurons.length;
  
  const wavePath = useMemo(() => {
    if (n < 2) return "";
    const points = neurons.map((neuron, i) => {
      const x = (i / (n - 1)) * width;
      const y = height / 2 - (neuron.V + 65) / 130 * (height - 20);
      return { x, y };
    });
    
    let d = `M ${points[0].x},${points[0].y}`;
    for (let i = 1; i < points.length; i++) {
      const cp1x = points[i-1].x + (points[i].x - points[i-1].x) / 3;
      const cp2x = points[i].x - (points[i].x - points[i-1].x) / 3;
      d += ` C ${cp1x},${points[i-1].y} ${cp2x},${points[i].y} ${points[i].x},${points[i].y}`;
    }
    return d;
  }, [neurons, width, height, n]);

  const forcePath = useMemo(() => {
    if (n < 2) return "";
    const points = neurons.map((neuron, i) => {
      const x = (i / (n - 1)) * width;
      const y = height - 10 - neuron.force * 60;
      return { x, y };
    });
    let d = `M ${points[0].x},${points[0].y}`;
    for (let i = 1; i < points.length; i++) {
      const cp1x = points[i-1].x + (points[i].x - points[i-1].x) / 3;
      const cp2x = points[i].x - (points[i].x - points[i-1].x) / 3;
      d += ` C ${cp1x},${points[i-1].y} ${cp2x},${points[i].y} ${points[i].x},${points[i].y}`;
    }
    return d;
  }, [neurons, width, height, n]);

  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      <rect width={width} height={height} fill={COLORS.bg} rx={6} />
      <line x1={0} y1={height/2} x2={width} y2={height/2} stroke={COLORS.textMuted} strokeWidth={0.5} strokeDasharray="2,4" />
      {wavePath && <path d={wavePath} fill="none" stroke={COLORS.accent2} strokeWidth={2} opacity={0.8} />}
      {forcePath && <path d={forcePath} fill="none" stroke={COLORS.accent4} strokeWidth={1.5} opacity={0.6} strokeDasharray="4,2" />}
      <text x={6} y={14} fontSize={11} fill={COLORS.textDim} fontFamily="'JetBrains Mono', monospace">WAVE PROPAGATION</text>
      <circle cx={width - 50} cy={10} r={3} fill={COLORS.accent2} />
      <text x={width - 44} y={13} fontSize={10} fill={COLORS.textDim} fontFamily="'JetBrains Mono', monospace">V_m</text>
      <circle cx={width - 25} cy={10} r={3} fill={COLORS.accent4} />
      <text x={width - 19} y={13} fontSize={10} fill={COLORS.textDim} fontFamily="'JetBrains Mono', monospace">F</text>
    </svg>
  );
}

function PhasePortrait({ V_hist, n_hist, width = 180, height = 180 }) {
  const tickCount = 5;
  const tickLen = 4;
  const points = useMemo(() => {
    if (!V_hist || V_hist.length < 2) return "";
    return V_hist.map((v, i) => {
      const x = ((v + 90) / 150) * (width - 20) + 10;
      const y = height - 10 - (n_hist[i] || 0) * (height - 20);
      return `${x},${y}`;
    }).join(" ");
  }, [V_hist, n_hist, width, height]);

  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      <defs>
        <filter id="pp-glow">
          <feGaussianBlur stdDeviation="4" result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
      </defs>
      <rect width={width} height={height} fill={COLORS.bgDeep} rx={6} />
      <line x1={10} y1={height - 10} x2={width - 10} y2={height - 10} stroke={COLORS.gridLine} strokeWidth={1} />
      <line x1={10} y1={10} x2={10} y2={height - 10} stroke={COLORS.gridLine} strokeWidth={1} />
      {Array.from({ length: tickCount }).map((_, i) => {
        const x = 10 + (i / (tickCount - 1)) * (width - 20);
        return <line key={`x-${i}`} x1={x} y1={height - 10} x2={x} y2={height - 10 - tickLen} stroke={COLORS.gridLine} strokeWidth={1} />;
      })}
      {Array.from({ length: tickCount }).map((_, i) => {
        const y = height - 10 - (i / (tickCount - 1)) * (height - 20);
        return <line key={`y-${i}`} x1={10} y1={y} x2={10 + tickLen} y2={y} stroke={COLORS.gridLine} strokeWidth={1} />;
      })}
      {points && <polyline points={points} fill="none" stroke={COLORS.accent3} strokeWidth={1.5} opacity={0.8} />}
      {V_hist && V_hist.length > 0 && (
        <circle 
          cx={((V_hist[V_hist.length-1] + 90) / 150) * (width - 20) + 10}
          cy={height - 10 - (n_hist[n_hist.length-1] || 0) * (height - 20)}
          r={5} fill={COLORS.accent3} filter="url(#pp-glow)" />
      )}
      <text x={width/2} y={height - 2} textAnchor="middle" fontSize={10} fill={COLORS.textDim} fontFamily="'JetBrains Mono', monospace">V (mV)</text>
      <text x={3} y={height/2} fontSize={10} fill={COLORS.textDim} fontFamily="'JetBrains Mono', monospace" transform={`rotate(-90, 8, ${height/2})`}>n gate</text>
      <text x={6} y={14} fontSize={11} fill={COLORS.accent3} fontFamily="'JetBrains Mono', monospace">PHASE PORTRAIT</text>
    </svg>
  );
}

function ParameterSlider({ label, value, min, max, step, onChange, color = COLORS.accent1, unit = "" }) {
  const pct = ((value - min) / (max - min)) * 100;
  const thumbSize = 14;
  const thumbOffset = `calc(${pct}% - ${thumbSize / 2}px)`;
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
        <span style={{ fontSize: 12, color: COLORS.textDim, fontFamily: "'JetBrains Mono', monospace" }}>{label}</span>
        <span style={{ fontSize: 12, color, fontFamily: "'JetBrains Mono', monospace", fontWeight: 600 }}>
          {typeof value === "number"
            ? (Math.abs(value) < 0.01 ? value.toFixed(6) : (Math.abs(value) < 1 ? value.toFixed(3) : value.toFixed(1)))
            : value}{unit}
        </span>
      </div>
      <div style={{ position: "relative", height: 16 }}>
        <div style={{
          position: "absolute", top: "50%", transform: "translateY(-50%)",
          width: "100%", height: 6, background: COLORS.textMuted, borderRadius: 999,
        }} />
        <div style={{
          position: "absolute", top: "50%", transform: "translateY(-50%)",
          width: `${pct}%`, height: 6, background: color, borderRadius: 999,
        }} />
        <div style={{
          position: "absolute", left: thumbOffset, top: "50%", transform: "translateY(-50%)",
          width: thumbSize, height: thumbSize, borderRadius: "50%",
          background: COLORS.bgDeep,
          border: `2px solid ${color}`,
          boxShadow: `0 0 10px ${color}66, 0 0 0 2px ${COLORS.bgDeep}`,
        }} />
        <input type="range" min={min} max={max} step={step} value={value} onChange={e => onChange(parseFloat(e.target.value))}
          style={{
            position: "absolute", inset: 0, width: "100%", height: 16, opacity: 0,
            appearance: "none", cursor: "pointer",
          }} />
      </div>
    </div>
  );
}

const clampPct = (value) => Math.max(0, Math.min(100, value));

const statusColor = (status) => {
  if (status === "NORMAL" || status === "BAL") return COLORS.accent1;
  if (status === "HIGH" || status === "EXCIT" || status === "TACHY") return COLORS.danger;
  if (status === "LOW" || status === "INHIB" || status === "BRADY") return COLORS.warning;
  return COLORS.textDim;
};

const PINN_PARAM_BOUNDS = {
  g_Na: [50, 200],
  g_K: [20, 80],
  g_Ca: [1, 10],
  omega_icc: [0.00015, 0.0008],
  coupling_strength: [0.05, 1.0],
};

const DRUG_LIBRARY = [
  {
    id: "mexiletine",
    name: "Mexiletine",
    className: "Na+ channel blocker",
    indication: "IBS-C / hyperexcitability",
    standardDoseMg: 200,
    maxDoseMg: 400,
    bioavailability: 0.85,
    halfLifeHours: 10.0,
    volumeDistributionL: 70.0,
    targets: [
      { key: "g_Na", fn: (baseline, conc) => baseline * (1 - 0.6 * conc / (conc + 10.0)) },
    ],
  },
  {
    id: "ondansetron",
    name: "Ondansetron",
    className: "5-HT3 antagonist",
    indication: "IBS-D / hypermotility",
    standardDoseMg: 8,
    maxDoseMg: 16,
    bioavailability: 0.6,
    halfLifeHours: 3.5,
    volumeDistributionL: 140.0,
    targets: [
      { key: "g_syn_e", fn: (baseline, conc) => baseline * (1 - 0.45 * conc / (conc + 2.0)) },
      { key: "omega_icc", fn: (baseline, conc) => baseline * (1 - 0.25 * conc / (conc + 2.0)) },
    ],
  },
  {
    id: "alosetron",
    name: "Alosetron",
    className: "Potent 5-HT3 antagonist",
    indication: "Severe IBS-D",
    standardDoseMg: 1,
    maxDoseMg: 2,
    bioavailability: 0.5,
    halfLifeHours: 1.5,
    volumeDistributionL: 65.0,
    targets: [
      { key: "g_syn_e", fn: (baseline, conc) => baseline * (1 - 0.6 * conc / (conc + 0.5)) },
      { key: "omega_icc", fn: (baseline, conc) => baseline * (1 - 0.35 * conc / (conc + 1.0)) },
    ],
  },
  {
    id: "lubiprostone",
    name: "Lubiprostone",
    className: "Cl- channel activator",
    indication: "IBS-C",
    standardDoseMg: 0.024,
    maxDoseMg: 0.048,
    bioavailability: 0.001,
    halfLifeHours: 0.9,
    volumeDistributionL: 100.0,
    targets: [
      { key: "g_L", fn: (baseline, conc) => baseline * (1 + 0.8 * conc / (conc + 5.0)) },
      { key: "omega_icc", fn: (baseline, conc) => baseline * (1 + 0.2 * conc / (conc + 10.0)) },
    ],
  },
  {
    id: "linaclotide",
    name: "Linaclotide",
    className: "GC-C agonist",
    indication: "IBS-C",
    standardDoseMg: 0.29,
    maxDoseMg: 0.58,
    bioavailability: 0.0,
    halfLifeHours: 0.1,
    volumeDistributionL: 1.0,
    targets: [
      { key: "coupling_strength", fn: (baseline, conc) => baseline * (1 + 0.5 * conc / (conc + 3.0)) },
      { key: "omega_icc", fn: (baseline, conc) => baseline * (1 + 0.3 * conc / (conc + 5.0)) },
    ],
  },
  {
    id: "prucalopride",
    name: "Prucalopride",
    className: "5-HT4 agonist",
    indication: "Constipation / prokinetic",
    standardDoseMg: 2,
    maxDoseMg: 4,
    bioavailability: 0.9,
    halfLifeHours: 24.0,
    volumeDistributionL: 567.0,
    targets: [
      { key: "g_syn_e", fn: (baseline, conc) => baseline * (1 + 0.6 * conc / (conc + 2.0)) },
      { key: "coupling_strength", fn: (baseline, conc) => baseline * (1 + 0.4 * conc / (conc + 3.0)) },
    ],
  },
  {
    id: "rifaximin",
    name: "Rifaximin",
    className: "Gut-selective antibiotic",
    indication: "IBS-D",
    standardDoseMg: 550,
    maxDoseMg: 1650,
    bioavailability: 0.004,
    halfLifeHours: 6.0,
    volumeDistributionL: 5.0,
    targets: [
      { key: "g_syn_e", fn: (baseline, conc) => baseline * (1 - 0.3 * conc / (conc + 10.0)) },
      { key: "g_syn_i", fn: (baseline, conc) => baseline * (1 + 0.2 * conc / (conc + 15.0)) },
    ],
  },
];

function utilClamp(value, minValue, maxValue) {
  return Math.max(minValue, Math.min(maxValue, value));
}

function utilMean(values) {
  if (!values || values.length === 0) return 0;
  return values.reduce((s, v) => s + v, 0) / values.length;
}

function utilVariance(values) {
  if (!values || values.length < 2) return 0;
  const mu = utilMean(values);
  return utilMean(values.map(v => (v - mu) ** 2));
}

function utilStd(values) {
  return Math.sqrt(utilVariance(values));
}

function utilCorrelation(a, b) {
  const n = Math.min(a.length, b.length);
  if (n < 3) return 0;
  const aa = a.slice(0, n);
  const bb = b.slice(0, n);
  const ma = utilMean(aa);
  const mb = utilMean(bb);
  let num = 0;
  let da = 0;
  let db = 0;
  for (let i = 0; i < n; i++) {
    const xa = aa[i] - ma;
    const xb = bb[i] - mb;
    num += xa * xb;
    da += xa * xa;
    db += xb * xb;
  }
  const denom = Math.sqrt(da * db) + 1e-12;
  return num / denom;
}

function parseNumericCsv(text) {
  const lines = text
    .split(/\r?\n/)
    .map(line => line.trim())
    .filter(Boolean);
  if (lines.length < 2) {
    throw new Error("CSV requires at least one header/data line and one data line.");
  }

  const splitLine = (line) => line.split(/[,\t; ]+/).filter(token => token.length > 0);
  const firstRow = splitLine(lines[0]);
  const hasHeader = firstRow.some(token => /[A-Za-z_]/.test(token));
  const headers = hasHeader ? firstRow : firstRow.map((_, idx) => `c${idx}`);
  const dataLines = hasHeader ? lines.slice(1) : lines;

  const rows = [];
  for (const line of dataLines) {
    const parts = splitLine(line);
    if (parts.length !== headers.length) continue;
    const nums = parts.map(v => Number(v));
    if (nums.every(Number.isFinite)) {
      rows.push(nums);
    }
  }
  if (rows.length < 2) {
    throw new Error("Could not parse numeric rows from CSV.");
  }
  return { headers, rows };
}

function estimateDominantFrequencyHz(timeSeconds, signalValues) {
  if (!timeSeconds || !signalValues || signalValues.length < 3) return 0;
  const n = Math.min(timeSeconds.length, signalValues.length);
  const signal = signalValues.slice(0, n);
  const time = timeSeconds.slice(0, n);
  const mu = utilMean(signal);
  let crossings = 0;
  for (let i = 1; i < n; i++) {
    const prev = signal[i - 1] - mu;
    const curr = signal[i] - mu;
    if (prev <= 0 && curr > 0) crossings += 1;
  }
  const duration = Math.max(1e-9, time[time.length - 1] - time[0]);
  return crossings / duration;
}

function summarizeSignalCsv(dataset, label) {
  const { headers, rows } = dataset;
  const nCols = rows[0]?.length || 0;
  if (nCols < 2) {
    throw new Error(`${label}: expected time + at least one channel.`);
  }

  const times = rows.map(r => r[0]);
  const durationSeconds = Math.max(1e-9, times[times.length - 1] - times[0]);
  const samplingHz = (rows.length - 1) / durationSeconds;
  const channelSeries = [];
  for (let c = 1; c < nCols; c++) {
    channelSeries.push(rows.map(r => r[c]));
  }

  const channelMeans = channelSeries.map(utilMean);
  const channelStds = channelSeries.map(utilStd);
  const allValues = channelSeries.flat();
  const avgCorr = channelSeries.length > 1
    ? utilMean(channelSeries.slice(1).map(ch => utilCorrelation(channelSeries[0], ch)))
    : 1.0;

  let dominantHz = estimateDominantFrequencyHz(times, channelSeries[0]);
  if (!Number.isFinite(dominantHz) || dominantHz <= 0 || durationSeconds < 20) {
    dominantHz = 0.05;
  }

  return {
    label,
    headers,
    rows,
    nSamples: rows.length,
    nChannels: nCols - 1,
    durationSeconds,
    samplingHz,
    mean: utilMean(allValues),
    std: utilStd(allValues),
    min: Math.min(...allValues),
    max: Math.max(...allValues),
    dominantHz,
    avgCorr,
    channelMeans,
    channelStds,
  };
}

function inferPatientParameters(patientSummary, fallbackParams = PARAMS) {
  if (!patientSummary?.egg) return null;

  const egg = patientSummary.egg;
  const hrm = patientSummary.hrm;
  const hrmScale = hrm ? utilClamp((hrm.mean - hrm.min) / ((hrm.max - hrm.min) + 1e-6), 0, 1) : 0.5;
  const hrmVar = hrm ? utilClamp(hrm.std / (Math.abs(hrm.mean) + 1e-6), 0, 1) : 0.25;
  const vmStd = utilClamp(egg.std, 0.1, 20);
  const vmMean = egg.mean;
  const corr = utilClamp(egg.avgCorr, -1, 1);
  const inferredHz = utilClamp(egg.dominantHz, 0.03, 0.09);

  const est = {
    g_Na: utilClamp(120 + (vmStd - 3.0) * 11 - (vmMean + 65) * 0.8, PINN_PARAM_BOUNDS.g_Na[0], PINN_PARAM_BOUNDS.g_Na[1]),
    g_K: utilClamp(36 + (3.0 - vmStd) * 6 + (1 - corr) * 4, PINN_PARAM_BOUNDS.g_K[0], PINN_PARAM_BOUNDS.g_K[1]),
    g_Ca: utilClamp(4 + hrmScale * 4 + hrmVar * 2.5, PINN_PARAM_BOUNDS.g_Ca[0], PINN_PARAM_BOUNDS.g_Ca[1]),
    omega_icc: utilClamp((2 * Math.PI * inferredHz) / 1000, PINN_PARAM_BOUNDS.omega_icc[0], PINN_PARAM_BOUNDS.omega_icc[1]),
    coupling_strength: utilClamp(0.25 + Math.max(0, corr) * 0.5, PINN_PARAM_BOUNDS.coupling_strength[0], PINN_PARAM_BOUNDS.coupling_strength[1]),
  };

  return {
    g_Na: { mean: est.g_Na, std: Math.max(1.5, est.g_Na * 0.03) },
    g_K: { mean: est.g_K, std: Math.max(0.8, est.g_K * 0.025) },
    g_Ca: { mean: est.g_Ca, std: Math.max(0.2, est.g_Ca * 0.03) },
    omega_icc: { mean: est.omega_icc, std: Math.max(0.00001, est.omega_icc * 0.06) },
    coupling_strength: { mean: est.coupling_strength, std: Math.max(0.01, est.coupling_strength * 0.08) },
    metadata: {
      fallbackOmegaUsed: egg.durationSeconds < 20,
      source: "frontend-surrogate",
      baselineOmega: fallbackParams.omega_icc,
    },
  };
}

function bootstrapParameterIntervals(patientSummary, fallbackParams = PARAMS, samples = 120) {
  const baseEstimate = inferPatientParameters(patientSummary, fallbackParams);
  if (!baseEstimate) return null;

  const records = [];
  for (let i = 0; i < samples; i++) {
    const jitter = (scale) => 1 + (Math.random() * 2 - 1) * scale;
    const jitteredSummary = {
      ...patientSummary,
      egg: {
        ...patientSummary.egg,
        mean: patientSummary.egg.mean * jitter(0.04),
        std: Math.max(0.05, patientSummary.egg.std * jitter(0.08)),
        avgCorr: utilClamp(patientSummary.egg.avgCorr * jitter(0.06), -1, 1),
        dominantHz: utilClamp(patientSummary.egg.dominantHz * jitter(0.2), 0.03, 0.09),
      },
      hrm: patientSummary.hrm ? {
        ...patientSummary.hrm,
        mean: patientSummary.hrm.mean * jitter(0.08),
        std: Math.max(0.05, patientSummary.hrm.std * jitter(0.1)),
      } : null,
    };
    records.push(inferPatientParameters(jitteredSummary, fallbackParams));
  }

  const keys = ["g_Na", "g_K", "g_Ca", "omega_icc", "coupling_strength"];
  const intervals = {};
  for (const key of keys) {
    const sorted = records.map(r => r[key].mean).sort((a, b) => a - b);
    const idx = (p) => sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor(sorted.length * p)))];
    intervals[key] = {
      p05: idx(0.05),
      p50: idx(0.5),
      p95: idx(0.95),
    };
  }
  return intervals;
}

function computePlasmaConcentration(doseMg, drug, timeHours = 2.0) {
  const molecularWeight = 300.0;
  const doseUmol = (doseMg / molecularWeight) * 1000.0;
  const cMax = (doseUmol * drug.bioavailability) / (drug.volumeDistributionL || 1);
  const halfLife = Math.max(0.01, drug.halfLifeHours || 0.01);
  const elimination = 0.693 / halfLife;
  return cMax * Math.exp(-elimination * Math.max(0, timeHours));
}

function applyDrugToParamSet(params, drug, doseMg, timeHours = 2.0) {
  const concentration = computePlasmaConcentration(doseMg, drug, timeHours);
  const out = { ...params };
  for (const target of drug.targets) {
    if (typeof out[target.key] === "number") {
      out[target.key] = target.fn(out[target.key], concentration);
    }
  }

  out.g_Na = utilClamp(out.g_Na, PINN_PARAM_BOUNDS.g_Na[0], PINN_PARAM_BOUNDS.g_Na[1]);
  out.g_K = utilClamp(out.g_K, PINN_PARAM_BOUNDS.g_K[0], PINN_PARAM_BOUNDS.g_K[1]);
  out.g_Ca = utilClamp(out.g_Ca, PINN_PARAM_BOUNDS.g_Ca[0], PINN_PARAM_BOUNDS.g_Ca[1]);
  out.omega_icc = utilClamp(out.omega_icc, PINN_PARAM_BOUNDS.omega_icc[0], PINN_PARAM_BOUNDS.omega_icc[1]);
  out.coupling_strength = utilClamp(out.coupling_strength, PINN_PARAM_BOUNDS.coupling_strength[0], PINN_PARAM_BOUNDS.coupling_strength[1]);
  out.g_syn_e = utilClamp(out.g_syn_e, 0.1, 3.0);
  out.g_syn_i = utilClamp(out.g_syn_i, 0.2, 5.0);

  return { params: out, concentration };
}

function quickBiomarkerModel(paramSet, profileId) {
  const iccCpm = (paramSet.omega_icc / (2 * Math.PI)) * 1000 * 60;
  const excitability = (paramSet.g_Na / (paramSet.g_K + 1)) * Math.sqrt(paramSet.g_Ca + 1) * (paramSet.g_syn_e / (paramSet.g_syn_i + 0.2));
  let motility = iccCpm * 10 + paramSet.g_Ca * 3 + paramSet.coupling_strength * 24 + paramSet.g_syn_e * 8 - paramSet.g_syn_i * 6;
  if (profileId === "hypersensitive") motility += 10;
  if (profileId === "hyposensitive") motility -= 10;
  if (profileId === "dysrhythmic") motility += (Math.random() - 0.5) * 14;

  return {
    icc_frequency_cpm: utilClamp(iccCpm, 0, 12),
    spike_rate_per_neuron: utilClamp((excitability - 3.2) * 1.8, 0, 12),
    motility_index: utilClamp(motility, 0, 100),
    propagation_correlation: utilClamp(paramSet.coupling_strength * 0.95 + 0.05, -1, 1),
  };
}

function approxErf(x) {
  const sign = x >= 0 ? 1 : -1;
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;
  const absX = Math.abs(x);
  const t = 1 / (1 + p * absX);
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-absX * absX);
  return sign * y;
}

function approxPValue(valsA, valsB) {
  const meanA = utilMean(valsA);
  const meanB = utilMean(valsB);
  const varA = utilVariance(valsA);
  const varB = utilVariance(valsB);
  const nA = Math.max(1, valsA.length);
  const nB = Math.max(1, valsB.length);
  const tLike = Math.abs(meanA - meanB) / Math.sqrt(varA / nA + varB / nB + 1e-12);
  const cdf = 0.5 * (1 + approxErf(tLike / Math.sqrt(2)));
  return utilClamp(2 * (1 - cdf), 0, 1);
}

function runVirtualDrugTrialFast({ baseParams, profileId, drug, dosesMg, cohortSize = 40 }) {
  const randomizeParams = () => {
    const vary = (v, rel = 0.15) => v * (1 + (Math.random() * 2 - 1) * rel);
    return {
      ...baseParams,
      g_Na: vary(baseParams.g_Na),
      g_K: vary(baseParams.g_K),
      g_Ca: vary(baseParams.g_Ca),
      g_syn_e: vary(baseParams.g_syn_e),
      g_syn_i: vary(baseParams.g_syn_i),
      omega_icc: vary(baseParams.omega_icc, 0.2),
      coupling_strength: vary(baseParams.coupling_strength, 0.2),
    };
  };

  const placebo = [];
  for (let i = 0; i < cohortSize; i++) {
    placebo.push(quickBiomarkerModel(randomizeParams(), profileId));
  }

  const doseResponse = {};
  let standardArm = null;
  for (const dose of dosesMg) {
    const treated = [];
    for (let i = 0; i < cohortSize; i++) {
      const randomized = randomizeParams();
      const treatedParams = applyDrugToParamSet(randomized, drug, dose).params;
      treated.push(quickBiomarkerModel(treatedParams, profileId));
    }
    if (Math.abs(dose - drug.standardDoseMg) < 1e-12) standardArm = treated;
    doseResponse[dose] = utilMean(treated.map(x => x.motility_index)) - utilMean(placebo.map(x => x.motility_index));
  }
  if (!standardArm) {
    const stdDose = drug.standardDoseMg;
    standardArm = [];
    for (let i = 0; i < cohortSize; i++) {
      const randomized = randomizeParams();
      const treatedParams = applyDrugToParamSet(randomized, drug, stdDose).params;
      standardArm.push(quickBiomarkerModel(treatedParams, profileId));
    }
  }

  const placeboMotility = placebo.map(x => x.motility_index);
  const treatedMotility = standardArm.map(x => x.motility_index);
  const placeboSpike = placebo.map(x => x.spike_rate_per_neuron);
  const treatedSpike = standardArm.map(x => x.spike_rate_per_neuron);
  const placeboCorr = placebo.map(x => x.propagation_correlation);
  const treatedCorr = standardArm.map(x => x.propagation_correlation);

  const responderRate = utilMean(standardArm.map((row, i) => {
    const base = Math.max(1e-3, placebo[i]?.motility_index || 1e-3);
    return ((row.motility_index - base) / base) > 0.3 ? 1 : 0;
  }));

  const effectSize = (a, b) => {
    const pooled = Math.sqrt((utilVariance(a) + utilVariance(b)) / 2 + 1e-12);
    return (utilMean(a) - utilMean(b)) / pooled;
  };

  const optimalDose = dosesMg.reduce((best, dose) => (
    (doseResponse[dose] > doseResponse[best] ? dose : best)
  ), dosesMg[0]);

  return {
    drugName: drug.name,
    profileId,
    cohortSize,
    dosesMg,
    doseResponse,
    optimalDoseMg: optimalDose,
    responderRate,
    pValues: {
      motility_index: approxPValue(treatedMotility, placeboMotility),
      spike_rate_per_neuron: approxPValue(treatedSpike, placeboSpike),
      propagation_correlation: approxPValue(treatedCorr, placeboCorr),
    },
    effectSizes: {
      motility_index: effectSize(treatedMotility, placeboMotility),
      spike_rate_per_neuron: effectSize(treatedSpike, placeboSpike),
      propagation_correlation: effectSize(treatedCorr, placeboCorr),
    },
    means: {
      placeboMotility: utilMean(placeboMotility),
      treatedMotility: utilMean(treatedMotility),
    },
  };
}

function generateSpicePreview(p) {
  return [
    "* ENS-GI Digital Twin - autogenerated preview netlist",
    ".title ENS Segment Chain",
    ".param gNa=" + p.g_Na.toFixed(4),
    ".param gK=" + p.g_K.toFixed(4),
    ".param gCa=" + p.g_Ca.toFixed(4),
    ".param gLeak=" + p.g_L.toFixed(4),
    ".param gCouple=" + p.coupling_strength.toFixed(4),
    ".param omega=" + p.omega_icc.toExponential(6),
    "",
    "* 12-cell chain (behavioral placeholder)",
    "Vstim nstim 0 PULSE(0 20u 5m 0.1m 0.1m 3m 20m)",
    "Rcouple0 n0 n1 {1/gCouple}",
    "Rcouple1 n1 n2 {1/gCouple}",
    "Rcouple2 n2 n3 {1/gCouple}",
    "* ... repeat for remaining segments ...",
    "Bicc nicc 0 V = 5*sin(2*pi*omega*time)",
    ".tran 0.05m 2s",
    ".end",
  ].join("\n");
}

function generateVerilogPreview(p) {
  return [
    "`include \"constants.vams\"",
    "`include \"disciplines.vams\"",
    "",
    "module ens_gi_segment(out, stim);",
    "  inout out, stim;",
    "  electrical out, stim;",
    "  parameter real g_Na = " + p.g_Na.toFixed(4) + ";",
    "  parameter real g_K = " + p.g_K.toFixed(4) + ";",
    "  parameter real g_Ca = " + p.g_Ca.toFixed(4) + ";",
    "  parameter real omega = " + p.omega_icc.toExponential(6) + ";",
    "  analog begin",
    "    I(out) <+ g_Na * tanh(V(stim));",
    "    I(out) <+ g_K * tanh(V(out));",
    "    I(out) <+ g_Ca * sin(2.0*`M_PI*omega*$abstime);",
    "  end",
    "endmodule",
  ].join("\n");
}

function StatusPill({ label, value, color = COLORS.text, accent = false }) {
  return (
    <div style={{
      minWidth: 100, padding: "6px 10px", borderRadius: 8,
      background: accent ? `${color}1a` : COLORS.bgDeep,
      border: `1px solid ${accent ? `${color}55` : COLORS.panelBorder}`,
      display: "flex", flexDirection: "column", gap: 2,
    }}>
      <div style={{ fontSize: 11, color: COLORS.textMicro, letterSpacing: "0.08em", textTransform: "uppercase" }}>
        {label}
      </div>
      <div style={{ fontSize: 14, color, fontWeight: 700 }}>
        {value}
      </div>
    </div>
  );
}

function SectionHeader({ layer, label, equation, color }) {
  return (
    <div style={{
      display: "flex", alignItems: "baseline", justifyContent: "space-between",
      borderBottom: `1px solid ${COLORS.divider}`, paddingBottom: 6, marginBottom: 10, gap: 12,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <div style={{
          padding: "2px 8px", borderRadius: 999, fontSize: 11, fontWeight: 700,
          color, background: `${color}22`, border: `1px solid ${color}66`,
        }}>
          {layer}
        </div>
        <div style={{ fontSize: 12, color, letterSpacing: "0.12em", textTransform: "uppercase" }}>
          {label}
        </div>
      </div>
      {equation && (
        <div style={{ fontSize: 11, color: COLORS.textDim, fontStyle: "italic" }}>
          {equation}
        </div>
      )}
    </div>
  );
}

function BiomarkerCard({ label, value, status, percent, color }) {
  const barColor = color || statusColor(status);
  return (
    <div style={{
      position: "relative", background: COLORS.bgDeep, borderRadius: 8, padding: "10px 10px 12px",
      border: `1px solid ${COLORS.panelBorder}`, overflow: "hidden",
    }}>
      <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 4, background: barColor }} />
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 8 }}>
        <div style={{ fontSize: 11, color: COLORS.textDim }}>{label}</div>
        <div style={{
          fontSize: 11, fontWeight: 700, color: barColor, textTransform: "uppercase",
          padding: "2px 6px", borderRadius: 999, border: `1px solid ${barColor}55`, background: `${barColor}18`,
        }}>
          {status}
        </div>
      </div>
      <div style={{ fontSize: 16, fontWeight: 700, color: COLORS.text, marginTop: 6 }}>
        {value}
      </div>
      <div style={{ marginTop: 8, height: 3, background: COLORS.panelBorder, borderRadius: 999, overflow: "hidden" }}>
        <div style={{
          width: `${clampPct(percent)}%`, height: "100%", background: barColor,
          transition: "width 0.6s ease",
        }} />
      </div>
    </div>
  );
}

// ─── Main App ───

const MODES = [
  { id: "research", label: "Research Simulator", color: COLORS.accent1 },
  { id: "neuromorphic", label: "Neuromorphic Model", color: COLORS.accent2 },
  { id: "clinical", label: "Clinical Predictor", color: COLORS.accent3 },
];

const IBS_PROFILES = [
  { id: "normal", label: "Healthy", desc: "Baseline motility and sensitivity", color: COLORS.accent2 },
  { id: "hypersensitive", label: "IBS-D (Hypersensitive)", desc: "Elevated excitability and pain sensitivity", color: COLORS.danger },
  { id: "hyposensitive", label: "IBS-C (Hyposensitive)", desc: "Reduced motility and neural drive", color: COLORS.warning },
  { id: "dysrhythmic", label: "IBS-M (Dysrhythmic)", desc: "Irregular pacing and coupling variability", color: COLORS.accent3 },
];

export default function ENSDigitalTwin() {
  const [mode, setMode] = useState("research");
  const [running, setRunning] = useState(false);
  const [ibsProfile, setIbsProfile] = useState("normal");
  const [stimNeuron, setStimNeuron] = useState(3);
  const [stimOn, setStimOn] = useState(false);
  const [stimCurrent, setStimCurrent] = useState(15);
  
  const [params, setParams] = useState({ ...PARAMS });
  const networkRef = useRef(null);
  const [tick, setTick] = useState(0);
  
  const [v0Hist, setV0Hist] = useState([]);
  const [n0Hist, setN0Hist] = useState([]);
  const [caHist, setCaHist] = useState([]);
  const [forceHist, setForceHist] = useState([]);
  const [iccHist, setIccHist] = useState([]);
  const [spikeRaster, setSpikeRaster] = useState([]);

  const [patientId, setPatientId] = useState("P001");
  const [patientData, setPatientData] = useState({ egg: null, hrm: null });
  const [patientSummary, setPatientSummary] = useState(null);
  const [patientError, setPatientError] = useState("");
  const [patientLoadStamp, setPatientLoadStamp] = useState("");
  const [estimateResult, setEstimateResult] = useState(null);
  const [bayesIntervals, setBayesIntervals] = useState(null);
  const [analysisBusy, setAnalysisBusy] = useState(false);
  const [epochs, setEpochs] = useState(80);
  const [syntheticSamples, setSyntheticSamples] = useState(200);
  const [bootstrapCount, setBootstrapCount] = useState(120);
  const [trialDrugId, setTrialDrugId] = useState("mexiletine");
  const [trialCohort, setTrialCohort] = useState(40);
  const [trialDosesText, setTrialDosesText] = useState("");
  const [trialBusy, setTrialBusy] = useState(false);
  const [trialResult, setTrialResult] = useState(null);
  const [hardwarePreview, setHardwarePreview] = useState({ type: "", text: "" });
  
  const HIST_LEN = 200;

  const initNetwork = useCallback((p = params) => {
    networkRef.current = new ENSNetwork(12, p);
    setV0Hist([]); setN0Hist([]); setCaHist([]); setForceHist([]); setIccHist([]);
    setSpikeRaster([]);
    setTick(0);
  }, [params]);

  useEffect(() => { initNetwork(); }, []);

  useEffect(() => {
    const profileParams = ibsProfile === "normal" ? {} : (IBS_PARAMS[ibsProfile] || {});
    const newParams = { ...PARAMS, ...profileParams };
    setParams(newParams);
    initNetwork(newParams);
  }, [ibsProfile]);

  useEffect(() => {
    if (!running) return;
    const interval = setInterval(() => {
      const net = networkRef.current;
      if (!net) return;
      
      for (let i = 0; i < 8; i++) {
        net.step(0.05, stimOn ? stimNeuron : -1, stimOn ? stimCurrent : 0);
      }
      
      const n0 = net.neurons[0];
      setV0Hist(prev => [...prev.slice(-(HIST_LEN - 1)), n0.V]);
      setN0Hist(prev => [...prev.slice(-(HIST_LEN - 1)), n0.n]);
      setCaHist(prev => [...prev.slice(-(HIST_LEN - 1)), n0.ca * 10000]);
      setForceHist(prev => [...prev.slice(-(HIST_LEN - 1)), n0.force * 100]);
      setIccHist(prev => [...prev.slice(-(HIST_LEN - 1)), Math.sin(net.icc_phase) * 5]);
      
      const spikes = net.neurons.map(n => n.spiked ? 1 : 0);
      setSpikeRaster(prev => [...prev.slice(-(HIST_LEN - 1)), spikes]);
      
      setTick(t => t + 1);
    }, 30);
    return () => clearInterval(interval);
  }, [running, stimOn, stimNeuron, stimCurrent]);

  const net = networkRef.current;
  const neurons = net ? net.neurons : [];
  const currentMode = MODES.find(m => m.id === mode);

  const avgV = neurons.length > 0 ? neurons.reduce((s, n) => s + n.V, 0) / neurons.length : -65;
  const recentSpikeFrames = spikeRaster.slice(-40);
  const activeNeuronSet = new Set();
  let recentSpikeEvents = 0;
  for (const spikes of recentSpikeFrames) {
    spikes.forEach((s, i) => {
      if (s) {
        activeNeuronSet.add(i);
        recentSpikeEvents += 1;
      }
    });
  }
  const spikeCount = activeNeuronSet.size;
  const avgForce = neurons.length > 0 ? neurons.reduce((s, n) => s + n.force, 0) / neurons.length : 0;
  const motilityIndex = (avgForce * 100).toFixed(1);
  const iccCpm = (params.omega_icc / (2 * Math.PI) * 1000 * 60);
  const eiRatio = params.g_syn_i ? params.g_syn_e / params.g_syn_i : 0;
  const caLoad = (neurons[0]?.ca || 0) * 10000;
  const styleTag = `
    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(1.4); }
    }
  `;

  const selectedDrug = useMemo(
    () => DRUG_LIBRARY.find(d => d.id === trialDrugId) || DRUG_LIBRARY[0],
    [trialDrugId]
  );

  const parsedTrialDoses = useMemo(() => {
    if (!selectedDrug) return [];
    if (!trialDosesText.trim()) return [selectedDrug.standardDoseMg];
    const values = trialDosesText
      .split(/[,\s]+/)
      .map(v => Number(v))
      .filter(v => Number.isFinite(v) && v > 0)
      .map(v => utilClamp(v, 0, selectedDrug.maxDoseMg));
    const uniqueSorted = Array.from(new Set(values)).sort((a, b) => a - b);
    return uniqueSorted.length ? uniqueSorted : [selectedDrug.standardDoseMg];
  }, [trialDosesText, selectedDrug]);

  const pinnCommand = useMemo(() => (
    `python scripts/train_pinn_from_patient_data.py --patient-id ${patientId} --segments 20 --epochs ${Math.round(epochs)} --synthetic-samples ${Math.round(syntheticSamples)} --bootstrap ${Math.round(bootstrapCount)} --model-out pinn_patient_model_${patientId.toLowerCase()}`
  ), [patientId, epochs, syntheticSamples, bootstrapCount]);

  const summarizePatient = useCallback((eggDataset, hrmDataset) => {
    if (!eggDataset && !hrmDataset) {
      setPatientSummary(null);
      return;
    }
    const summary = {
      egg: eggDataset ? summarizeSignalCsv(eggDataset, "EGG") : null,
      hrm: hrmDataset ? summarizeSignalCsv(hrmDataset, "HRM") : null,
    };
    setPatientSummary(summary);
  }, []);

  const handlePatientFile = useCallback(async (kind, file) => {
    if (!file) return;
    try {
      const text = await file.text();
      const parsed = parseNumericCsv(text);
      setPatientData(prev => {
        const next = { ...prev, [kind]: parsed };
        summarizePatient(next.egg, next.hrm);
        return next;
      });
      setPatientLoadStamp(`${kind.toUpperCase()} loaded: ${file.name}`);
      setPatientError("");
    } catch (err) {
      setPatientError(`Could not parse ${kind.toUpperCase()} CSV: ${err?.message || err}`);
    }
  }, [summarizePatient]);

  const loadDemoPatient = useCallback(async (id) => {
    setPatientId(id);
    try {
      const loadCsv = async (suffix) => {
        const res = await fetch(`/patient_data/${id}_${suffix}.csv`, { cache: "no-store" });
        if (!res.ok) return null;
        return parseNumericCsv(await res.text());
      };

      const [egg, hrm] = await Promise.all([loadCsv("egg"), loadCsv("hrm")]);
      if (!egg && !hrm) {
        throw new Error(`No demo files found for ${id}.`);
      }

      setPatientData({ egg, hrm });
      summarizePatient(egg, hrm);
      setPatientLoadStamp(`Demo patient ${id} loaded.`);
      setPatientError("");
    } catch (err) {
      setPatientError(`Demo load failed: ${err?.message || err}`);
    }
  }, [summarizePatient]);

  const applyEstimatedParamsToModel = useCallback((estimate) => {
    if (!estimate) return;
    const nextParams = {
      ...params,
      g_Na: estimate.g_Na?.mean ?? params.g_Na,
      g_K: estimate.g_K?.mean ?? params.g_K,
      g_Ca: estimate.g_Ca?.mean ?? params.g_Ca,
      omega_icc: estimate.omega_icc?.mean ?? params.omega_icc,
      coupling_strength: estimate.coupling_strength?.mean ?? params.coupling_strength,
    };
    setParams(nextParams);
    const liveNet = networkRef.current;
    if (liveNet) {
      liveNet.params.omega_icc = nextParams.omega_icc;
      liveNet.params.coupling_strength = nextParams.coupling_strength;
      liveNet.neurons.forEach(n => {
        n.p.g_Na = nextParams.g_Na;
        n.p.g_K = nextParams.g_K;
        n.p.g_Ca = nextParams.g_Ca;
        n.p.omega_icc = nextParams.omega_icc;
        n.p.coupling_strength = nextParams.coupling_strength;
      });
    }
  }, [params]);

  const runFrontendPinn = useCallback(() => {
    if (!patientSummary?.egg) {
      setPatientError("Load at least an EGG CSV before PINN estimation.");
      return;
    }
    setAnalysisBusy(true);
    setTimeout(() => {
      const estimate = inferPatientParameters(patientSummary, params);
      setEstimateResult(estimate);
      setBayesIntervals(null);
      setAnalysisBusy(false);
      setPatientError("");
    }, 30);
  }, [patientSummary, params]);

  const runBayesianBootstrap = useCallback(() => {
    if (!patientSummary?.egg) {
      setPatientError("Load patient data before Bayesian uncertainty estimation.");
      return;
    }
    setAnalysisBusy(true);
    setTimeout(() => {
      const intervals = bootstrapParameterIntervals(
        patientSummary,
        params,
        Math.round(utilClamp(bootstrapCount, 30, 1000))
      );
      setBayesIntervals(intervals);
      setAnalysisBusy(false);
      setPatientError("");
    }, 30);
  }, [patientSummary, params, bootstrapCount]);

  const runDrugTrial = useCallback(() => {
    if (!selectedDrug) return;
    setTrialBusy(true);
    setTimeout(() => {
      const result = runVirtualDrugTrialFast({
        baseParams: params,
        profileId: ibsProfile,
        drug: selectedDrug,
        dosesMg: parsedTrialDoses,
        cohortSize: Math.round(utilClamp(trialCohort, 10, 500)),
      });
      setTrialResult(result);
      setTrialBusy(false);
    }, 30);
  }, [params, ibsProfile, selectedDrug, parsedTrialDoses, trialCohort]);

  const generateHardwareArtifact = useCallback((kind) => {
    if (kind === "spice") {
      setHardwarePreview({ type: "SPICE", text: generateSpicePreview(params) });
    } else {
      setHardwarePreview({ type: "Verilog-A", text: generateVerilogPreview(params) });
    }
  }, [params]);

  return (
    <div style={{
      width: "100%", minHeight: "100vh", background: COLORS.bg, color: COLORS.text,
      fontFamily: "'JetBrains Mono', 'SF Mono', 'Fira Code', monospace",
      padding: "20px 20px 32px 20px", boxSizing: "border-box",
      borderTop: running ? `3px solid ${COLORS.accent1}` : `3px solid ${COLORS.panelBorder}`,
      transition: "border-color 0.2s ease",
    }}>
      <style>{styleTag}</style>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16, flexWrap: "wrap", gap: 8 }}>
        <div>
          <div style={{ fontSize: 11, color: COLORS.textDim, letterSpacing: "0.2em", textTransform: "uppercase", marginBottom: 2 }}>
            Multiscale Simulation Engine
          </div>
          <div style={{ fontSize: 20, fontWeight: 700, letterSpacing: "-0.02em" }}>
            <span style={{ color: COLORS.accent1 }}>ENS</span>
            <span style={{ color: COLORS.textDim }}>–</span>
            <span style={{ color: COLORS.accent2 }}>GI</span>
            <span style={{ color: COLORS.textDim }}> Digital Twin</span>
          </div>
        </div>
        
        <div style={{ display: "flex", gap: 6, borderBottom: `1px solid ${COLORS.panelBorder}` }}>
          {MODES.map(m => (
            <button key={m.id} onClick={() => setMode(m.id)}
              style={{
                padding: "8px 12px",
                border: `1px solid ${mode === m.id ? m.color : COLORS.panelBorder}`,
                borderBottom: mode === m.id ? `2px solid ${m.color}` : `1px solid ${COLORS.panelBorder}`,
                background: mode === m.id ? COLORS.panelHover : "transparent",
                color: mode === m.id ? COLORS.text : COLORS.textDim,
                borderRadius: "6px 6px 0 0", cursor: "pointer", fontSize: 12, fontFamily: "inherit",
                fontWeight: mode === m.id ? 700 : 600, transition: "all 0.2s",
              }}>
              {m.label}
            </button>
          ))}
        </div>
      </div>

      {/* Status Bar */}
      <div style={{
        display: "flex", gap: 12, marginBottom: 12, padding: "10px 12px",
        background: COLORS.panel, borderRadius: 10, border: `1px solid ${COLORS.panelBorder}`,
        flexWrap: "wrap", alignItems: "center",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <button onClick={() => setRunning(!running)} style={{
            padding: "10px 16px",
            background: running ? `${COLORS.danger}22` : `${COLORS.accent1}22`,
            color: running ? COLORS.danger : COLORS.accent1,
            border: `1px solid ${running ? COLORS.danger : COLORS.accent1}`,
            borderRadius: 10, cursor: "pointer",
            fontSize: 12, fontWeight: 700, fontFamily: "inherit",
            display: "flex", alignItems: "center", gap: 8,
            boxShadow: running ? `0 0 16px ${COLORS.runGlow}` : "none",
          }}>
            <span style={{
              width: 8, height: 8, borderRadius: "50%",
              background: running ? COLORS.accent1 : COLORS.textMuted,
              boxShadow: running ? `0 0 8px ${COLORS.runGlow}` : "none",
              animation: running ? "pulse 1.2s ease-in-out infinite" : "none",
            }} />
            {running ? "STOP" : "RUN"}
          </button>
        </div>
        <div style={{ display: "flex", flex: 1, gap: 8, flexWrap: "wrap" }}>
          <StatusPill label="Time" value={`${net ? net.time.toFixed(1) : "0.0"} ms`} color={COLORS.accent2} />
          <StatusPill label="Neurons" value={neurons.length} color={COLORS.accent1} />
          <StatusPill label="Active" value={`${spikeCount}/${neurons.length || 0}`} color={spikeCount > 0 ? COLORS.accent1 : COLORS.textDim} accent={spikeCount > 0} />
          <StatusPill label="Spikes" value={recentSpikeEvents} color={recentSpikeEvents > 0 ? COLORS.accent2 : COLORS.textDim} accent={recentSpikeEvents > 0} />
          <StatusPill label="Motility" value={`${motilityIndex}%`} color={COLORS.accent4} />
          <StatusPill label="Mode" value={currentMode.label.toUpperCase()} color={currentMode.color} accent />
        </div>
        <div style={{ marginLeft: "auto" }}>
          <button onClick={() => initNetwork()} style={{
            padding: "8px 12px", background: "transparent", border: `1px solid ${COLORS.textMuted}`,
            color: COLORS.textDim, borderRadius: 6, cursor: "pointer", fontSize: 12, fontFamily: "inherit",
          }}>
            RESET
          </button>
        </div>
      </div>

      {/* Main Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 300px", gap: 12 }}>
        {/* Left: Visualizations */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          
          {/* Layer 1: Electrophysiology */}
          <div style={{
            background: COLORS.panelL1, borderRadius: 8, border: `1px solid ${COLORS.panelL1Border}`,
            borderLeft: `3px solid ${COLORS.accent1}`, padding: 12,
          }}>
            <SectionHeader
              layer="L1"
              label="Cellular Electrophysiology"
              equation="C_m dV/dt = -Sum I_ion + I_syn + I_stim"
              color={COLORS.accent1}
            />
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              <MiniScope data={v0Hist} color={COLORS.accent1} width={240} height={90} label="V_m (Neuron 0)" value={neurons[0]?.V} unit="mV" />
              <MiniScope data={caHist} color={COLORS.accent3} width={200} height={90} label="[Ca²⁺]ᵢ" value={neurons[0]?.ca * 10000} unit="nM" />
              <PhasePortrait V_hist={v0Hist} n_hist={n0Hist} width={200} height={200} />
            </div>
          </div>

          {/* Layer 2: Network */}
          <div style={{
            background: COLORS.panelL2, borderRadius: 8, border: `1px solid ${COLORS.panelL2Border}`,
            borderLeft: `3px solid ${COLORS.accent2}`, padding: 12,
          }}>
            <SectionHeader
              layer="L2"
              label="ENS Network & Propagation"
              color={COLORS.accent2}
            />
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              <NeuronGrid neurons={neurons} width={380} height={140} />
              <div style={{ flex: 1, minWidth: 200 }}>
                <PropagationWave neurons={neurons} width={240} height={120} />
              </div>
            </div>
            
            {/* Spike Raster */}
            <div style={{ marginTop: 8 }}>
              <div style={{ fontSize: 11, color: COLORS.textDim, letterSpacing: "0.08em", marginBottom: 6 }}>SPIKE RASTER</div>
              <svg width="100%" height={72} viewBox={`0 0 ${HIST_LEN} 60`} preserveAspectRatio="none"
                style={{ display: "block", borderRadius: 6, border: `1px solid ${COLORS.panelBorder}` }}>
                <rect width={HIST_LEN} height={60} fill={COLORS.bgDeep} />
                {spikeRaster.map((spikes, t) =>
                  spikes.map((s, n) => s ? (
                    <rect key={`${t}-${n}`} x={t} y={n * 5} width={1} height={4} fill={COLORS.accent2} opacity={0.8} />
                  ) : null)
                )}
              </svg>
            </div>
          </div>

          {/* Layer 3: ICC & Motility */}
          <div style={{
            background: COLORS.panelL3, borderRadius: 8, border: `1px solid ${COLORS.panelL3Border}`,
            borderLeft: `3px solid ${COLORS.accent4}`, padding: 12,
          }}>
            <SectionHeader
              layer="L3"
              label="ICC Pacemaker & Motility"
              color={COLORS.accent4}
            />
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              <MiniScope data={iccHist} color={COLORS.warning} label="ICC Slow Wave" value={net ? Math.sin(net.icc_phase) * 5 : 0} unit="mV" />
              <MiniScope data={forceHist} color={COLORS.accent4} label="Contractile Force" value={avgForce * 100} unit="%" />
              
              {/* Gut tube visualization */}
              <svg width={140} height={65} style={{ display: "block" }}>
                <rect width={140} height={65} fill={COLORS.bg} rx={4} />
                <text x={4} y={12} fontSize={10} fill={COLORS.textDim} fontFamily="'JetBrains Mono', monospace">GUT TUBE</text>
                {neurons.map((n, i) => {
                  const x = 10 + (i / (neurons.length - 1)) * 120;
                  const radius = 8 + n.force * 12;
                  return (
                    <ellipse key={i} cx={x} cy={38} rx={5} ry={radius}
                      fill="none" stroke={COLORS.accent4} strokeWidth={1}
                      opacity={0.3 + n.force * 0.7} />
                  );
                })}
              </svg>
            </div>
          </div>

          {/* Mode-specific panel */}
          {mode === "neuromorphic" && (
            <div style={{ background: COLORS.panel, borderRadius: 8, border: `1px solid ${COLORS.accent2}33`, padding: 12 }}>
              <div style={{ fontSize: 11, color: COLORS.accent2, letterSpacing: "0.15em", marginBottom: 8 }}>
                ⚡ NEUROMORPHIC MAPPING — CIRCUIT EQUIVALENTS
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 8 }}>
                {[
                  { bio: "ENS Neuron", hw: "RC + NL Cond.", param: `g_Na=${params.g_Na}` },
                  { bio: "Ion Channel", hw: "Verilog-A Module", param: `HH gates` },
                  { bio: "Synapse", hw: "CCCS + τ filter", param: `τ=${params.tau_syn}ms` },
                  { bio: "ICC Oscillator", hw: "Wien Bridge", param: `f=${iccCpm.toFixed(2)} cpm` },
                  { bio: "Gap Junction", hw: "Resistor Chain", param: `g=${params.coupling_strength}` },
                  { bio: "Smooth Muscle", hw: "E-M Transducer", param: `k_ca=${params.k_ca.toExponential(2)}` },
                ].map((item, i) => (
                  <div key={i} style={{ background: COLORS.bgDeep, borderRadius: 6, padding: 10, border: `1px solid ${COLORS.panelBorder}` }}>
                    <div style={{ fontSize: 12, color: COLORS.accent2, fontWeight: 600 }}>{item.bio}</div>
                    <div style={{ fontSize: 11, color: COLORS.textDim }}>→ {item.hw}</div>
                    <div style={{ fontSize: 11, color: COLORS.accent1, marginTop: 2 }}>{item.param}</div>
                  </div>
                ))}
              </div>
              <div style={{ marginTop: 10, display: "flex", gap: 8, flexWrap: "wrap" }}>
                <button onClick={() => generateHardwareArtifact("spice")} style={{
                  padding: "6px 10px", borderRadius: 6, border: `1px solid ${COLORS.accent2}`,
                  background: `${COLORS.accent2}1a`, color: COLORS.accent2, fontFamily: "inherit", cursor: "pointer", fontSize: 12,
                }}>
                  Generate SPICE Preview
                </button>
                <button onClick={() => generateHardwareArtifact("verilog")} style={{
                  padding: "6px 10px", borderRadius: 6, border: `1px solid ${COLORS.accent1}`,
                  background: `${COLORS.accent1}1a`, color: COLORS.accent1, fontFamily: "inherit", cursor: "pointer", fontSize: 12,
                }}>
                  Generate Verilog-A Preview
                </button>
              </div>
              {hardwarePreview.text && (
                <div style={{ marginTop: 8 }}>
                  <div style={{ fontSize: 11, color: COLORS.textDim, marginBottom: 6 }}>
                    {hardwarePreview.type} output preview
                  </div>
                  <textarea
                    readOnly
                    value={hardwarePreview.text}
                    style={{
                      width: "100%",
                      minHeight: 160,
                      resize: "vertical",
                      background: COLORS.bgDeep,
                      color: COLORS.text,
                      border: `1px solid ${COLORS.panelBorder}`,
                      borderRadius: 6,
                      padding: 8,
                      fontFamily: "'JetBrains Mono', monospace",
                      fontSize: 11,
                    }}
                  />
                </div>
              )}
            </div>
          )}

          {mode === "clinical" && (
            <div style={{ background: COLORS.panel, borderRadius: 8, border: `1px solid ${COLORS.accent3}33`, padding: 12 }}>
              <div style={{ fontSize: 11, color: COLORS.accent3, letterSpacing: "0.15em", marginBottom: 8 }}>
                CLINICAL WORKBENCH - {IBS_PROFILES.find(p => p.id === ibsProfile)?.label.toUpperCase()}
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 10 }}>
                {[
                  { label: "Mean Vm", value: `${avgV.toFixed(1)} mV`, status: avgV > -50 ? "HIGH" : avgV < -70 ? "LOW" : "NORMAL", percent: ((avgV + 80) / 40) * 100 },
                  { label: "Spike Rate", value: `${spikeCount}/12`, status: spikeCount > 6 ? "HIGH" : spikeCount < 2 ? "LOW" : "NORMAL", percent: (spikeCount / 12) * 100 },
                  { label: "Motility Index", value: `${motilityIndex}%`, status: parseFloat(motilityIndex) > 50 ? "HIGH" : parseFloat(motilityIndex) < 10 ? "LOW" : "NORMAL", percent: parseFloat(motilityIndex) },
                  { label: "ICC Frequency", value: `${iccCpm.toFixed(2)} cpm`, status: params.omega_icc > 0.00045 ? "TACHY" : params.omega_icc < 0.0002 ? "BRADY" : "NORMAL", percent: ((params.omega_icc - 0.00015) / (0.0008 - 0.00015)) * 100 },
                  { label: "E/I Balance", value: `${eiRatio.toFixed(2)}`, status: eiRatio > 0.8 ? "EXCIT" : eiRatio < 0.3 ? "INHIB" : "BAL", percent: (eiRatio / 1.5) * 100 },
                  { label: "Ca Load", value: `${caLoad.toFixed(1)} nM`, status: (neurons[0]?.ca || 0) > 0.001 ? "HIGH" : "NORMAL", percent: (caLoad / 20) * 100 },
                ].map((item, i) => (
                  <BiomarkerCard key={i} label={item.label} value={item.value} status={item.status} percent={item.percent} />
                ))}
              </div>

              <div style={{ marginTop: 12, display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 10 }}>
                <div style={{ background: COLORS.bgDeep, border: `1px solid ${COLORS.panelBorder}`, borderRadius: 8, padding: 10 }}>
                  <div style={{ fontSize: 11, color: COLORS.accent3, letterSpacing: "0.08em", marginBottom: 8 }}>PATIENT DATA INGEST</div>
                  <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 8 }}>
                    {["P001", "P002", "P003"].map(id => (
                      <button key={id} onClick={() => loadDemoPatient(id)} style={{
                        padding: "4px 8px", borderRadius: 4, border: `1px solid ${COLORS.panelBorder}`,
                        background: id === patientId ? `${COLORS.accent2}1a` : "transparent",
                        color: id === patientId ? COLORS.accent2 : COLORS.textDim, cursor: "pointer", fontSize: 11, fontFamily: "inherit",
                      }}>
                        Demo {id}
                      </button>
                    ))}
                  </div>
                  <div style={{ display: "grid", gap: 6 }}>
                    <label style={{ fontSize: 11, color: COLORS.textDim }}>
                      EGG CSV
                      <input type="file" accept=".csv,text/csv" onChange={e => handlePatientFile("egg", e.target.files?.[0])}
                        style={{ display: "block", width: "100%", marginTop: 4, fontSize: 11 }} />
                    </label>
                    <label style={{ fontSize: 11, color: COLORS.textDim }}>
                      HRM CSV
                      <input type="file" accept=".csv,text/csv" onChange={e => handlePatientFile("hrm", e.target.files?.[0])}
                        style={{ display: "block", width: "100%", marginTop: 4, fontSize: 11 }} />
                    </label>
                  </div>
                  {patientLoadStamp && <div style={{ marginTop: 8, fontSize: 11, color: COLORS.accent1 }}>{patientLoadStamp}</div>}
                  {patientError && <div style={{ marginTop: 8, fontSize: 11, color: COLORS.danger }}>{patientError}</div>}
                  {patientSummary && (
                    <div style={{ marginTop: 8, fontSize: 11, color: COLORS.textDim, lineHeight: 1.5 }}>
                      {patientSummary.egg && <div>EGG: {patientSummary.egg.nSamples} samples, {patientSummary.egg.nChannels} channels, fs={patientSummary.egg.samplingHz.toFixed(1)} Hz</div>}
                      {patientSummary.hrm && <div>HRM: {patientSummary.hrm.nSamples} samples, {patientSummary.hrm.nChannels} channels, fs={patientSummary.hrm.samplingHz.toFixed(1)} Hz</div>}
                    </div>
                  )}
                </div>

                <div style={{ background: COLORS.bgDeep, border: `1px solid ${COLORS.panelBorder}`, borderRadius: 8, padding: 10 }}>
                  <div style={{ fontSize: 11, color: COLORS.accent2, letterSpacing: "0.08em", marginBottom: 8 }}>PINN + BAYESIAN CONTROLS</div>
                  <label style={{ fontSize: 10, color: COLORS.textDim, display: "block", marginBottom: 6 }}>
                    Patient ID
                    <input value={patientId} onChange={e => setPatientId(e.target.value || "P001")}
                      style={{ width: "100%", marginTop: 3, background: COLORS.panel, color: COLORS.text, border: `1px solid ${COLORS.panelBorder}`, borderRadius: 4, padding: "4px 6px" }} />
                  </label>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(70px, 1fr))", gap: 6 }}>
                    <label style={{ fontSize: 10, color: COLORS.textDim }}>
                      Epochs
                      <input type="number" min={1} value={epochs} onChange={e => setEpochs(Number(e.target.value) || 1)}
                        style={{ width: "100%", marginTop: 3, background: COLORS.panel, color: COLORS.text, border: `1px solid ${COLORS.panelBorder}`, borderRadius: 4, padding: "4px 6px" }} />
                    </label>
                    <label style={{ fontSize: 10, color: COLORS.textDim }}>
                      Synth
                      <input type="number" min={10} value={syntheticSamples} onChange={e => setSyntheticSamples(Number(e.target.value) || 10)}
                        style={{ width: "100%", marginTop: 3, background: COLORS.panel, color: COLORS.text, border: `1px solid ${COLORS.panelBorder}`, borderRadius: 4, padding: "4px 6px" }} />
                    </label>
                    <label style={{ fontSize: 10, color: COLORS.textDim }}>
                      Bootstrap
                      <input type="number" min={30} value={bootstrapCount} onChange={e => setBootstrapCount(Number(e.target.value) || 30)}
                        style={{ width: "100%", marginTop: 3, background: COLORS.panel, color: COLORS.text, border: `1px solid ${COLORS.panelBorder}`, borderRadius: 4, padding: "4px 6px" }} />
                    </label>
                  </div>
                  <div style={{ display: "flex", gap: 6, marginTop: 8, flexWrap: "wrap" }}>
                    <button onClick={runFrontendPinn} disabled={analysisBusy} style={{
                      padding: "6px 8px", borderRadius: 6, border: `1px solid ${COLORS.accent2}`,
                      background: `${COLORS.accent2}18`, color: COLORS.accent2, cursor: "pointer", fontSize: 11, fontFamily: "inherit",
                    }}>
                      {analysisBusy ? "Estimating..." : "Estimate Parameters"}
                    </button>
                    <button onClick={runBayesianBootstrap} disabled={analysisBusy} style={{
                      padding: "6px 8px", borderRadius: 6, border: `1px solid ${COLORS.warning}`,
                      background: `${COLORS.warning}18`, color: COLORS.warning, cursor: "pointer", fontSize: 11, fontFamily: "inherit",
                    }}>
                      Bayesian CI
                    </button>
                    <button onClick={() => applyEstimatedParamsToModel(estimateResult)} disabled={!estimateResult} style={{
                      padding: "6px 8px", borderRadius: 6, border: `1px solid ${COLORS.accent1}`,
                      background: `${COLORS.accent1}18`, color: COLORS.accent1, cursor: "pointer", fontSize: 11, fontFamily: "inherit",
                    }}>
                      Apply to Model
                    </button>
                  </div>
                  <div style={{ marginTop: 8, fontSize: 10, color: COLORS.textDim, whiteSpace: "pre-wrap", wordBreak: "break-all" }}>
                    Backend command: {pinnCommand}
                  </div>
                  {estimateResult && (
                    <div style={{ marginTop: 8, fontSize: 11, color: COLORS.textDim, lineHeight: 1.55 }}>
                      {["g_Na", "g_K", "g_Ca", "omega_icc", "coupling_strength"].map(key => (
                        <div key={key}>
                          {key}: {estimateResult[key].mean.toFixed(key === "omega_icc" ? 6 : 3)} +/- {estimateResult[key].std.toFixed(key === "omega_icc" ? 6 : 3)}
                        </div>
                      ))}
                    </div>
                  )}
                  {bayesIntervals && (
                    <div style={{ marginTop: 8, fontSize: 11, color: COLORS.textDim, lineHeight: 1.55 }}>
                      {["g_Na", "g_K", "g_Ca", "omega_icc", "coupling_strength"].map(key => (
                        <div key={key}>
                          {key} 90% CI: [{bayesIntervals[key].p05.toFixed(key === "omega_icc" ? 6 : 3)}, {bayesIntervals[key].p95.toFixed(key === "omega_icc" ? 6 : 3)}]
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                <div style={{ background: COLORS.bgDeep, border: `1px solid ${COLORS.panelBorder}`, borderRadius: 8, padding: 10 }}>
                  <div style={{ fontSize: 11, color: COLORS.accent4, letterSpacing: "0.08em", marginBottom: 8 }}>VIRTUAL DRUG TRIALS</div>
                  <div style={{ display: "grid", gap: 6 }}>
                    <label style={{ fontSize: 11, color: COLORS.textDim }}>
                      Drug
                      <select value={trialDrugId} onChange={e => setTrialDrugId(e.target.value)} style={{
                        width: "100%", marginTop: 4, background: COLORS.panel, color: COLORS.text, border: `1px solid ${COLORS.panelBorder}`, borderRadius: 4, padding: "4px 6px",
                      }}>
                        {DRUG_LIBRARY.map(drug => (
                          <option key={drug.id} value={drug.id}>{drug.name} ({drug.className})</option>
                        ))}
                      </select>
                    </label>
                    <label style={{ fontSize: 11, color: COLORS.textDim }}>
                      Doses mg (comma separated)
                      <input value={trialDosesText} onChange={e => setTrialDosesText(e.target.value)}
                        placeholder={`${selectedDrug?.standardDoseMg ?? ""}`}
                        style={{ width: "100%", marginTop: 4, background: COLORS.panel, color: COLORS.text, border: `1px solid ${COLORS.panelBorder}`, borderRadius: 4, padding: "4px 6px" }} />
                    </label>
                    <label style={{ fontSize: 11, color: COLORS.textDim }}>
                      Cohort size
                      <input type="number" min={10} max={500} value={trialCohort} onChange={e => setTrialCohort(Number(e.target.value) || 10)}
                        style={{ width: "100%", marginTop: 4, background: COLORS.panel, color: COLORS.text, border: `1px solid ${COLORS.panelBorder}`, borderRadius: 4, padding: "4px 6px" }} />
                    </label>
                  </div>
                  <button onClick={runDrugTrial} disabled={trialBusy} style={{
                    marginTop: 8, width: "100%", padding: "6px 8px", borderRadius: 6, border: `1px solid ${COLORS.accent4}`,
                    background: `${COLORS.accent4}1a`, color: COLORS.accent4, cursor: "pointer", fontSize: 11, fontFamily: "inherit",
                  }}>
                    {trialBusy ? "Running..." : "Run Virtual Trial"}
                  </button>
                  {trialResult && (
                    <div style={{ marginTop: 8, fontSize: 11, color: COLORS.textDim, lineHeight: 1.55 }}>
                      <div>Optimal dose: {trialResult.optimalDoseMg} mg</div>
                      <div>Responder rate: {(trialResult.responderRate * 100).toFixed(1)}%</div>
                      <div>Motility p-value: {trialResult.pValues.motility_index.toExponential(2)}</div>
                      <div style={{ marginTop: 6 }}>
                        {Object.entries(trialResult.doseResponse).map(([dose, delta]) => (
                          <div key={dose} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
                            <span style={{ width: 54 }}>{dose} mg</span>
                            <div style={{ flex: 1, background: COLORS.panelBorder, height: 6, borderRadius: 999, overflow: "hidden" }}>
                              <div style={{
                                width: `${utilClamp(Math.abs(delta) * 2, 0, 100)}%`,
                                height: "100%",
                                background: delta >= 0 ? COLORS.accent1 : COLORS.danger,
                              }} />
                            </div>
                            <span style={{ width: 55, textAlign: "right", color: delta >= 0 ? COLORS.accent1 : COLORS.danger }}>
                              {delta >= 0 ? "+" : ""}{delta.toFixed(2)}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right: Controls */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          
          {/* Stimulation */}
          <div style={{ background: COLORS.panel, borderRadius: 8, border: `1px solid ${COLORS.panelBorder}`, padding: 12 }}>
            <div style={{
              fontSize: 11, color: COLORS.textDim, letterSpacing: "0.1em",
              marginBottom: 10, paddingBottom: 6, borderBottom: `1px solid ${COLORS.divider}`,
            }}>STIMULATION</div>
            <button onClick={() => setStimOn(!stimOn)} style={{
              width: "100%", padding: "6px", marginBottom: 8,
              background: stimOn ? `${COLORS.warning}20` : "transparent",
              border: `1px solid ${stimOn ? COLORS.warning : COLORS.textMuted}`,
              color: stimOn ? COLORS.warning : COLORS.textDim,
              borderRadius: 4, cursor: "pointer", fontSize: 12, fontFamily: "inherit", fontWeight: 600,
            }}>
              {stimOn ? `⚡ STIM ON (${stimCurrent.toFixed(0)} uA)` : "○ STIM OFF"}
            </button>
            <ParameterSlider label="Target Neuron" value={stimNeuron} min={0} max={11} step={1}
              onChange={setStimNeuron} color={COLORS.warning} />
            <ParameterSlider label="Stim Current" value={stimCurrent} min={0} max={40} step={1}
              onChange={setStimCurrent} color={COLORS.warning} unit=" uA" />
          </div>

          {/* IBS Profile */}
          <div style={{ background: COLORS.panel, borderRadius: 8, border: `1px solid ${COLORS.panelBorder}`, padding: 12 }}>
            <div style={{
              fontSize: 11, color: COLORS.textDim, letterSpacing: "0.1em",
              marginBottom: 10, paddingBottom: 6, borderBottom: `1px solid ${COLORS.divider}`,
            }}>IBS PROFILE</div>
            {IBS_PROFILES.map(p => {
              const isActive = ibsProfile === p.id;
              return (
                <button key={p.id} onClick={() => setIbsProfile(p.id)} style={{
                  width: "100%", padding: "8px 10px", marginBottom: 6, textAlign: "left",
                  background: isActive ? `${p.color}1a` : COLORS.bgDeep,
                  border: `1px solid ${isActive ? p.color : COLORS.panelBorder}`,
                  borderLeft: `4px solid ${p.color}`,
                  color: isActive ? p.color : COLORS.text,
                  borderRadius: 6, cursor: "pointer", fontSize: 12, fontFamily: "inherit",
                }}>
                  <div style={{ fontWeight: 600 }}>{p.label}</div>
                  <div style={{ fontSize: 11, color: COLORS.textDim, marginTop: 2 }}>{p.desc}</div>
                </button>
              );
            })}
          </div>

          {/* Parameters */}
          <div style={{ background: COLORS.panel, borderRadius: 8, border: `1px solid ${COLORS.panelBorder}`, padding: 12 }}>
            <div style={{
              fontSize: 11, color: COLORS.textDim, letterSpacing: "0.1em",
              marginBottom: 10, paddingBottom: 6, borderBottom: `1px solid ${COLORS.divider}`,
            }}>MEMBRANE PARAMS</div>
            <ParameterSlider label="g_Na" value={params.g_Na} min={20} max={200} step={5}
              onChange={v => { setParams(p => ({...p, g_Na: v})); if(net) net.neurons.forEach(n => n.p.g_Na = v); }}
              color={COLORS.accent1} unit=" mS/cm²" />
            <ParameterSlider label="g_K" value={params.g_K} min={10} max={80} step={2}
              onChange={v => { setParams(p => ({...p, g_K: v})); if(net) net.neurons.forEach(n => n.p.g_K = v); }}
              color={COLORS.accent2} unit=" mS/cm²" />
            <ParameterSlider label="g_Ca" value={params.g_Ca} min={0} max={15} step={0.5}
              onChange={v => { setParams(p => ({...p, g_Ca: v})); if(net) net.neurons.forEach(n => n.p.g_Ca = v); }}
              color={COLORS.accent3} unit=" mS/cm²" />
          </div>

          <div style={{ background: COLORS.panel, borderRadius: 8, border: `1px solid ${COLORS.panelBorder}`, padding: 12 }}>
            <div style={{
              fontSize: 11, color: COLORS.textDim, letterSpacing: "0.1em",
              marginBottom: 10, paddingBottom: 6, borderBottom: `1px solid ${COLORS.divider}`,
            }}>NETWORK PARAMS</div>
            <ParameterSlider label="Coupling" value={params.coupling_strength} min={0} max={2} step={0.05}
              onChange={v => { setParams(p => ({...p, coupling_strength: v})); if(net) net.params.coupling_strength = v; }}
              color={COLORS.accent2} />
            <ParameterSlider label="omega_ICC" value={params.omega_icc} min={0.00015} max={0.0008} step={0.00001}
              onChange={v => { setParams(p => ({...p, omega_icc: v})); if(net) net.params.omega_icc = v; }}
              color={COLORS.warning} unit=" rad/ms" />
            <ParameterSlider label="g_syn_e" value={params.g_syn_e} min={0} max={3} step={0.1}
              onChange={v => { setParams(p => ({...p, g_syn_e: v})); if(net) net.neurons.forEach(n => n.p.g_syn_e = v); }}
              color={COLORS.accent1} unit=" mS" />
            <ParameterSlider label="g_syn_i" value={params.g_syn_i} min={0} max={5} step={0.1}
              onChange={v => { setParams(p => ({...p, g_syn_i: v})); if(net) net.neurons.forEach(n => n.p.g_syn_i = v); }}
              color={COLORS.danger} unit=" mS" />
          </div>

          {/* Architecture info */}
          <div style={{ background: COLORS.panel, borderRadius: 8, border: `1px solid ${COLORS.panelBorder}`, padding: 12 }}>
            <div style={{
              fontSize: 11, color: COLORS.textDim, letterSpacing: "0.1em",
              marginBottom: 10, paddingBottom: 6, borderBottom: `1px solid ${COLORS.divider}`,
            }}>ARCHITECTURE</div>
            <div style={{ fontSize: 10, color: COLORS.textDim, lineHeight: 1.6 }}>
              <div><span style={{ color: COLORS.accent1 }}>L1</span> HH + Ca electrophysiology</div>
              <div><span style={{ color: COLORS.accent2 }}>L2</span> Coupled ENS network (E/I)</div>
              <div><span style={{ color: COLORS.accent4 }}>L3</span> ICC pacemaker -&gt; motility</div>
              <div><span style={{ color: COLORS.accent3 }}>L4</span> Clinical AI and virtual trials</div>
              <div style={{ marginTop: 4, borderTop: `1px solid ${COLORS.panelBorder}`, paddingTop: 4 }}>
                <span style={{ color: COLORS.accent1 }}>Frontend</span> -&gt; Live simulator<br/>
                <span style={{ color: COLORS.accent2 }}>Frontend</span> -&gt; SPICE/Verilog preview<br/>
                <span style={{ color: COLORS.accent3 }}>Frontend</span> -&gt; Patient ingest + PINN/Bayes + drug trial
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}


