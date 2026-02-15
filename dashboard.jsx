import { useState, useEffect, useRef, useCallback, useMemo } from "react";

// ─── Simulation Engine (JavaScript port of core ENS-GI model) ───
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
  omega_icc: 0.05, // ~3 cpm
  A_icc: 5.0,
  // Smooth muscle
  k_ca: 0.0002, tau_ca: 50.0,  // Unified with Python: faster Ca2+ clearance
  Ca_half_force: 0.02,          // Ca2+ for half-maximal force (mM)
  hill_n: 3.0,                  // Hill coefficient (cooperativity)
  // Network coupling
  coupling_strength: 0.3,
};

const IBS_PARAMS = {
  hypersensitive: { g_Na: 150, g_Ca: 6.0, g_syn_e: 0.8, omega_icc: 0.08 },
  hyposensitive: { g_Na: 80, g_K: 50, g_syn_i: 2.0, omega_icc: 0.03 },
  dysrhythmic: { g_Ca: 8.0, omega_icc: 0.12, coupling_strength: 0.1 },
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
    this.icc_phase += this.params.omega_icc * dt;
    
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
  panel: "#111827",
  panelBorder: "#1e293b",
  accent1: "#00e5a0",
  accent2: "#00b4d8",
  accent3: "#e040fb",
  accent4: "#ff6b35",
  text: "#e2e8f0",
  textDim: "#64748b",
  textMuted: "#334155",
  danger: "#ef4444",
  warning: "#f59e0b",
  gridLine: "#1a2332",
};

function MiniScope({ data, color, width = 200, height = 60, label, value, unit }) {
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
    <div style={{ position: "relative" }}>
      <svg width={width} height={height} style={{ display: "block" }}>
        <rect width={width} height={height} fill={COLORS.bg} rx={4} />
        <line x1={0} y1={height/2} x2={width} y2={height/2} stroke={COLORS.textMuted} strokeWidth={0.5} strokeDasharray="3,3" />
        {points && <polyline points={points} fill="none" stroke={color} strokeWidth={1.5} opacity={0.9} />}
      </svg>
      {label && (
        <div style={{ position: "absolute", top: 3, left: 6, fontSize: 9, color: COLORS.textDim, fontFamily: "'JetBrains Mono', monospace", letterSpacing: "0.05em" }}>
          {label}
        </div>
      )}
      {value !== undefined && (
        <div style={{ position: "absolute", bottom: 3, right: 6, fontSize: 10, color, fontFamily: "'JetBrains Mono', monospace", fontWeight: 600 }}>
          {typeof value === "number" ? value.toFixed(1) : value}{unit && <span style={{ fontSize: 8, opacity: 0.7 }}> {unit}</span>}
        </div>
      )}
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
            <text x={x} y={height - 1} textAnchor="middle" fontSize={7} fill={COLORS.textDim} fontFamily="'JetBrains Mono', monospace">
              {i}
            </text>
          </g>
        );
      })}
      
      <text x={6} y={12} fontSize={9} fill={COLORS.textDim} fontFamily="'JetBrains Mono', monospace">
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
      <text x={6} y={12} fontSize={9} fill={COLORS.textDim} fontFamily="'JetBrains Mono', monospace">WAVE PROPAGATION</text>
      <circle cx={width - 50} cy={10} r={3} fill={COLORS.accent2} />
      <text x={width - 44} y={13} fontSize={8} fill={COLORS.textDim} fontFamily="'JetBrains Mono', monospace">V_m</text>
      <circle cx={width - 25} cy={10} r={3} fill={COLORS.accent4} />
      <text x={width - 19} y={13} fontSize={8} fill={COLORS.textDim} fontFamily="'JetBrains Mono', monospace">F</text>
    </svg>
  );
}

function PhasePortrait({ V_hist, n_hist, width = 180, height = 180 }) {
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
      <rect width={width} height={height} fill={COLORS.bg} rx={6} />
      {points && <polyline points={points} fill="none" stroke={COLORS.accent3} strokeWidth={1} opacity={0.7} />}
      {V_hist && V_hist.length > 0 && (
        <circle 
          cx={((V_hist[V_hist.length-1] + 90) / 150) * (width - 20) + 10}
          cy={height - 10 - (n_hist[n_hist.length-1] || 0) * (height - 20)}
          r={3} fill={COLORS.accent3} />
      )}
      <text x={width/2} y={height - 2} textAnchor="middle" fontSize={8} fill={COLORS.textDim} fontFamily="'JetBrains Mono', monospace">V (mV)</text>
      <text x={3} y={height/2} fontSize={8} fill={COLORS.textDim} fontFamily="'JetBrains Mono', monospace" transform={`rotate(-90, 8, ${height/2})`}>n gate</text>
      <text x={6} y={12} fontSize={9} fill={COLORS.textDim} fontFamily="'JetBrains Mono', monospace">PHASE PORTRAIT</text>
    </svg>
  );
}

function ParameterSlider({ label, value, min, max, step, onChange, color = COLORS.accent1, unit = "" }) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
        <span style={{ fontSize: 10, color: COLORS.textDim, fontFamily: "'JetBrains Mono', monospace" }}>{label}</span>
        <span style={{ fontSize: 10, color, fontFamily: "'JetBrains Mono', monospace", fontWeight: 600 }}>
          {typeof value === "number" ? (value < 1 ? value.toFixed(3) : value.toFixed(1)) : value}{unit}
        </span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={e => onChange(parseFloat(e.target.value))}
        style={{ width: "100%", height: 4, appearance: "none", background: `linear-gradient(to right, ${color} ${pct}%, ${COLORS.textMuted} ${pct}%)`,
          borderRadius: 2, outline: "none", cursor: "pointer" }} />
    </div>
  );
}

// ─── Main App ───

const MODES = [
  { id: "research", label: "Research Simulator", icon: "🔬", color: COLORS.accent1 },
  { id: "neuromorphic", label: "Neuromorphic Model", icon: "⚡", color: COLORS.accent2 },
  { id: "clinical", label: "Clinical Predictor", icon: "🏥", color: COLORS.accent3 },
];

const IBS_PROFILES = [
  { id: "normal", label: "Healthy" },
  { id: "hypersensitive", label: "IBS-D (Hypersensitive)" },
  { id: "hyposensitive", label: "IBS-C (Hyposensitive)" },
  { id: "dysrhythmic", label: "IBS-M (Dysrhythmic)" },
];

export default function ENSDigitalTwin() {
  const [mode, setMode] = useState("research");
  const [running, setRunning] = useState(false);
  const [ibsProfile, setIbsProfile] = useState("normal");
  const [stimNeuron, setStimNeuron] = useState(3);
  const [stimOn, setStimOn] = useState(false);
  
  const [params, setParams] = useState({ ...PARAMS });
  const networkRef = useRef(null);
  const [tick, setTick] = useState(0);
  
  const [v0Hist, setV0Hist] = useState([]);
  const [n0Hist, setN0Hist] = useState([]);
  const [caHist, setCaHist] = useState([]);
  const [forceHist, setForceHist] = useState([]);
  const [iccHist, setIccHist] = useState([]);
  const [spikeRaster, setSpikeRaster] = useState([]);
  
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
        net.step(0.05, stimOn ? stimNeuron : -1, 15);
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
  }, [running, stimOn, stimNeuron]);

  const net = networkRef.current;
  const neurons = net ? net.neurons : [];
  const currentMode = MODES.find(m => m.id === mode);

  const avgV = neurons.length > 0 ? neurons.reduce((s, n) => s + n.V, 0) / neurons.length : -65;
  const spikeCount = neurons.filter(n => n.V > -20).length;
  const avgForce = neurons.length > 0 ? neurons.reduce((s, n) => s + n.force, 0) / neurons.length : 0;
  const motilityIndex = (avgForce * 100).toFixed(1);

  return (
    <div style={{
      width: "100%", minHeight: "100vh", background: COLORS.bg, color: COLORS.text,
      fontFamily: "'JetBrains Mono', 'SF Mono', 'Fira Code', monospace",
      padding: 16, boxSizing: "border-box",
    }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16, flexWrap: "wrap", gap: 8 }}>
        <div>
          <div style={{ fontSize: 10, color: COLORS.textDim, letterSpacing: "0.2em", textTransform: "uppercase", marginBottom: 2 }}>
            Multiscale Simulation Engine
          </div>
          <div style={{ fontSize: 20, fontWeight: 700, letterSpacing: "-0.02em" }}>
            <span style={{ color: COLORS.accent1 }}>ENS</span>
            <span style={{ color: COLORS.textDim }}>–</span>
            <span style={{ color: COLORS.accent2 }}>GI</span>
            <span style={{ color: COLORS.textDim }}> Digital Twin</span>
          </div>
        </div>
        
        <div style={{ display: "flex", gap: 4 }}>
          {MODES.map(m => (
            <button key={m.id} onClick={() => setMode(m.id)}
              style={{
                padding: "6px 12px", border: `1px solid ${mode === m.id ? m.color : COLORS.panelBorder}`,
                background: mode === m.id ? `${m.color}15` : "transparent",
                color: mode === m.id ? m.color : COLORS.textDim,
                borderRadius: 6, cursor: "pointer", fontSize: 10, fontFamily: "inherit",
                transition: "all 0.2s",
              }}>
              {m.icon} {m.label}
            </button>
          ))}
        </div>
      </div>

      {/* Status Bar */}
      <div style={{
        display: "flex", gap: 16, marginBottom: 12, padding: "8px 12px",
        background: COLORS.panel, borderRadius: 8, border: `1px solid ${COLORS.panelBorder}`,
        flexWrap: "wrap", alignItems: "center",
      }}>
        <div style={{ fontSize: 10 }}>
          <span style={{ color: COLORS.textDim }}>MODE </span>
          <span style={{ color: currentMode.color, fontWeight: 600 }}>{currentMode.label.toUpperCase()}</span>
        </div>
        <div style={{ fontSize: 10 }}>
          <span style={{ color: COLORS.textDim }}>t = </span>
          <span style={{ color: COLORS.text }}>{net ? net.time.toFixed(1) : "0.0"} ms</span>
        </div>
        <div style={{ fontSize: 10 }}>
          <span style={{ color: COLORS.textDim }}>NEURONS </span>
          <span style={{ color: COLORS.accent1 }}>{neurons.length}</span>
        </div>
        <div style={{ fontSize: 10 }}>
          <span style={{ color: COLORS.textDim }}>ACTIVE </span>
          <span style={{ color: spikeCount > 0 ? COLORS.accent1 : COLORS.textDim }}>{spikeCount}</span>
        </div>
        <div style={{ fontSize: 10 }}>
          <span style={{ color: COLORS.textDim }}>MOTILITY </span>
          <span style={{ color: COLORS.accent4 }}>{motilityIndex}%</span>
        </div>
        <div style={{ flex: 1 }} />
        <button onClick={() => setRunning(!running)} style={{
          padding: "4px 16px", background: running ? COLORS.danger : COLORS.accent1,
          color: COLORS.bg, border: "none", borderRadius: 4, cursor: "pointer",
          fontSize: 10, fontWeight: 700, fontFamily: "inherit",
        }}>
          {running ? "■ STOP" : "▶ RUN"}
        </button>
        <button onClick={() => initNetwork()} style={{
          padding: "4px 12px", background: "transparent", border: `1px solid ${COLORS.textMuted}`,
          color: COLORS.textDim, borderRadius: 4, cursor: "pointer", fontSize: 10, fontFamily: "inherit",
        }}>
          ↺ RESET
        </button>
      </div>

      {/* Main Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 220px", gap: 12 }}>
        {/* Left: Visualizations */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          
          {/* Layer 1: Electrophysiology */}
          <div style={{ background: COLORS.panel, borderRadius: 8, border: `1px solid ${COLORS.panelBorder}`, padding: 12 }}>
            <div style={{ fontSize: 10, color: COLORS.accent1, letterSpacing: "0.15em", marginBottom: 8, display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{ width: 6, height: 6, borderRadius: "50%", background: COLORS.accent1, display: "inline-block" }} />
              LAYER 1 — CELLULAR ELECTROPHYSIOLOGY
              <span style={{ fontSize: 9, color: COLORS.textDim, fontStyle: "italic", marginLeft: 4 }}>
                C_m dV/dt = -ΣI_ion + I_syn + I_stim
              </span>
            </div>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              <MiniScope data={v0Hist} color={COLORS.accent1} width={220} height={70} label="V_m (Neuron 0)" value={neurons[0]?.V} unit="mV" />
              <MiniScope data={caHist} color={COLORS.accent3} width={160} height={70} label="[Ca²⁺]ᵢ" value={neurons[0]?.ca * 10000} unit="nM" />
              <PhasePortrait V_hist={v0Hist} n_hist={n0Hist} width={140} height={70} />
            </div>
          </div>

          {/* Layer 2: Network */}
          <div style={{ background: COLORS.panel, borderRadius: 8, border: `1px solid ${COLORS.panelBorder}`, padding: 12 }}>
            <div style={{ fontSize: 10, color: COLORS.accent2, letterSpacing: "0.15em", marginBottom: 8, display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{ width: 6, height: 6, borderRadius: "50%", background: COLORS.accent2, display: "inline-block" }} />
              LAYER 2 — ENS NETWORK & PROPAGATION
            </div>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              <NeuronGrid neurons={neurons} width={300} height={120} />
              <div style={{ flex: 1, minWidth: 200 }}>
                <PropagationWave neurons={neurons} width={240} height={120} />
              </div>
            </div>
            
            {/* Spike Raster */}
            <div style={{ marginTop: 8 }}>
              <svg width="100%" height={60} viewBox={`0 0 ${HIST_LEN} 60`} preserveAspectRatio="none" style={{ display: "block", borderRadius: 4 }}>
                <rect width={HIST_LEN} height={60} fill={COLORS.bg} />
                {spikeRaster.map((spikes, t) =>
                  spikes.map((s, n) => s ? (
                    <rect key={`${t}-${n}`} x={t} y={n * 5} width={1} height={4} fill={COLORS.accent2} opacity={0.8} />
                  ) : null)
                )}
                <text x={2} y={8} fontSize={4} fill={COLORS.textDim}>SPIKE RASTER</text>
              </svg>
            </div>
          </div>

          {/* Layer 3: ICC & Motility */}
          <div style={{ background: COLORS.panel, borderRadius: 8, border: `1px solid ${COLORS.panelBorder}`, padding: 12 }}>
            <div style={{ fontSize: 10, color: COLORS.accent4, letterSpacing: "0.15em", marginBottom: 8, display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{ width: 6, height: 6, borderRadius: "50%", background: COLORS.accent4, display: "inline-block" }} />
              LAYER 3 — ICC PACEMAKER & MOTILITY
            </div>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              <MiniScope data={iccHist} color={COLORS.warning} width={200} height={65} label="ICC Slow Wave" value={net ? Math.sin(net.icc_phase) * 5 : 0} unit="mV" />
              <MiniScope data={forceHist} color={COLORS.accent4} width={200} height={65} label="Contractile Force" value={avgForce * 100} unit="%" />
              
              {/* Gut tube visualization */}
              <svg width={140} height={65} style={{ display: "block" }}>
                <rect width={140} height={65} fill={COLORS.bg} rx={4} />
                <text x={4} y={10} fontSize={8} fill={COLORS.textDim} fontFamily="'JetBrains Mono', monospace">GUT TUBE</text>
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
              <div style={{ fontSize: 10, color: COLORS.accent2, letterSpacing: "0.15em", marginBottom: 8 }}>
                ⚡ NEUROMORPHIC MAPPING — CIRCUIT EQUIVALENTS
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 8 }}>
                {[
                  { bio: "ENS Neuron", hw: "RC + NL Cond.", param: `g_Na=${params.g_Na}` },
                  { bio: "Ion Channel", hw: "Verilog-A Module", param: `HH gates` },
                  { bio: "Synapse", hw: "CCCS + τ filter", param: `τ=${params.tau_syn}ms` },
                  { bio: "ICC Oscillator", hw: "Wien Bridge", param: `f=${(params.omega_icc/(2*Math.PI)*1000).toFixed(1)}cpm` },
                  { bio: "Gap Junction", hw: "Resistor Chain", param: `g=${params.coupling_strength}` },
                  { bio: "Smooth Muscle", hw: "E→M Transducer", param: `k=${params.k_force}` },
                ].map((item, i) => (
                  <div key={i} style={{ background: COLORS.bg, borderRadius: 4, padding: 8, border: `1px solid ${COLORS.panelBorder}` }}>
                    <div style={{ fontSize: 10, color: COLORS.accent2, fontWeight: 600 }}>{item.bio}</div>
                    <div style={{ fontSize: 9, color: COLORS.textDim }}>→ {item.hw}</div>
                    <div style={{ fontSize: 8, color: COLORS.accent1, marginTop: 2 }}>{item.param}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {mode === "clinical" && (
            <div style={{ background: COLORS.panel, borderRadius: 8, border: `1px solid ${COLORS.accent3}33`, padding: 12 }}>
              <div style={{ fontSize: 10, color: COLORS.accent3, letterSpacing: "0.15em", marginBottom: 8 }}>
                🏥 CLINICAL BIOMARKERS — {IBS_PROFILES.find(p => p.id === ibsProfile)?.label.toUpperCase()}
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))", gap: 8 }}>
                {[
                  { label: "Mean Vm", value: `${avgV.toFixed(1)} mV`, status: avgV > -50 ? "HIGH" : avgV < -70 ? "LOW" : "NORMAL" },
                  { label: "Spike Rate", value: `${spikeCount}/12`, status: spikeCount > 6 ? "HIGH" : spikeCount < 2 ? "LOW" : "NORMAL" },
                  { label: "Motility Index", value: `${motilityIndex}%`, status: parseFloat(motilityIndex) > 50 ? "HIGH" : parseFloat(motilityIndex) < 10 ? "LOW" : "NORMAL" },
                  { label: "ICC Frequency", value: `${(params.omega_icc / (2 * Math.PI) * 1000 * 60).toFixed(1)} cpm`, status: params.omega_icc > 0.08 ? "TACHY" : params.omega_icc < 0.04 ? "BRADY" : "NORMAL" },
                  { label: "E/I Balance", value: `${(params.g_syn_e / params.g_syn_i).toFixed(2)}`, status: params.g_syn_e / params.g_syn_i > 0.8 ? "EXCIT" : params.g_syn_e / params.g_syn_i < 0.3 ? "INHIB" : "BAL" },
                  { label: "Ca²⁺ Load", value: `${(neurons[0]?.ca * 10000 || 0).toFixed(1)} nM`, status: (neurons[0]?.ca || 0) > 0.001 ? "HIGH" : "NORMAL" },
                ].map((item, i) => (
                  <div key={i} style={{ background: COLORS.bg, borderRadius: 4, padding: 8, border: `1px solid ${COLORS.panelBorder}` }}>
                    <div style={{ fontSize: 8, color: COLORS.textDim, marginBottom: 2 }}>{item.label}</div>
                    <div style={{ fontSize: 13, color: COLORS.text, fontWeight: 600 }}>{item.value}</div>
                    <div style={{
                      fontSize: 8, fontWeight: 700, marginTop: 2,
                      color: item.status === "NORMAL" || item.status === "BAL" ? COLORS.accent1 :
                             item.status === "HIGH" || item.status === "EXCIT" || item.status === "TACHY" ? COLORS.danger : COLORS.warning,
                    }}>
                      {item.status}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right: Controls */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          
          {/* Stimulation */}
          <div style={{ background: COLORS.panel, borderRadius: 8, border: `1px solid ${COLORS.panelBorder}`, padding: 12 }}>
            <div style={{ fontSize: 10, color: COLORS.textDim, letterSpacing: "0.1em", marginBottom: 8 }}>STIMULATION</div>
            <button onClick={() => setStimOn(!stimOn)} style={{
              width: "100%", padding: "6px", marginBottom: 8,
              background: stimOn ? `${COLORS.warning}20` : "transparent",
              border: `1px solid ${stimOn ? COLORS.warning : COLORS.textMuted}`,
              color: stimOn ? COLORS.warning : COLORS.textDim,
              borderRadius: 4, cursor: "pointer", fontSize: 10, fontFamily: "inherit", fontWeight: 600,
            }}>
              {stimOn ? "⚡ STIM ON" : "○ STIM OFF"}
            </button>
            <ParameterSlider label="Target Neuron" value={stimNeuron} min={0} max={11} step={1}
              onChange={setStimNeuron} color={COLORS.warning} />
          </div>

          {/* IBS Profile */}
          <div style={{ background: COLORS.panel, borderRadius: 8, border: `1px solid ${COLORS.panelBorder}`, padding: 12 }}>
            <div style={{ fontSize: 10, color: COLORS.textDim, letterSpacing: "0.1em", marginBottom: 8 }}>IBS PROFILE</div>
            {IBS_PROFILES.map(p => (
              <button key={p.id} onClick={() => setIbsProfile(p.id)} style={{
                width: "100%", padding: "5px 8px", marginBottom: 4, textAlign: "left",
                background: ibsProfile === p.id ? `${COLORS.accent3}15` : "transparent",
                border: `1px solid ${ibsProfile === p.id ? COLORS.accent3 : COLORS.panelBorder}`,
                color: ibsProfile === p.id ? COLORS.accent3 : COLORS.textDim,
                borderRadius: 4, cursor: "pointer", fontSize: 9, fontFamily: "inherit",
              }}>
                {ibsProfile === p.id ? "● " : "○ "}{p.label}
              </button>
            ))}
          </div>

          {/* Parameters */}
          <div style={{ background: COLORS.panel, borderRadius: 8, border: `1px solid ${COLORS.panelBorder}`, padding: 12 }}>
            <div style={{ fontSize: 10, color: COLORS.textDim, letterSpacing: "0.1em", marginBottom: 8 }}>MEMBRANE PARAMS</div>
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
            <div style={{ fontSize: 10, color: COLORS.textDim, letterSpacing: "0.1em", marginBottom: 8 }}>NETWORK PARAMS</div>
            <ParameterSlider label="Coupling" value={params.coupling_strength} min={0} max={2} step={0.05}
              onChange={v => { setParams(p => ({...p, coupling_strength: v})); if(net) net.params.coupling_strength = v; }}
              color={COLORS.accent2} />
            <ParameterSlider label="ω_ICC" value={params.omega_icc} min={0.01} max={0.2} step={0.005}
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
            <div style={{ fontSize: 10, color: COLORS.textDim, letterSpacing: "0.1em", marginBottom: 6 }}>ARCHITECTURE</div>
            <div style={{ fontSize: 8, color: COLORS.textDim, lineHeight: 1.6 }}>
              <div><span style={{ color: COLORS.accent1 }}>L1</span> HH + Ca²⁺ electrophysiology</div>
              <div><span style={{ color: COLORS.accent2 }}>L2</span> Coupled ENS network (E/I)</div>
              <div><span style={{ color: COLORS.accent4 }}>L3</span> ICC pacemaker → motility</div>
              <div style={{ marginTop: 4, borderTop: `1px solid ${COLORS.panelBorder}`, paddingTop: 4 }}>
                <span style={{ color: COLORS.accent1 }}>Python</span> → Research sim<br/>
                <span style={{ color: COLORS.accent2 }}>SPICE</span> → Hardware model<br/>
                <span style={{ color: COLORS.accent3 }}>ML+Opt</span> → Clinical tool
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
