# Verilog-A Standard Cell Library

**ENS-GI Digital Twin — Hardware Module Library**

This directory contains Verilog-A behavioral models for ENS (Enteric Nervous System) cellular components. These modules can be used in:
- Cadence Virtuoso / Spectre
- Keysight ADS
- Synopsys HSPICE
- Other Verilog-A compatible simulators

---

## Available Modules

### Ion Channels

| Module | File | Description | Parameters |
|--------|------|-------------|----------|
| **NaV1.5** | `NaV1_5.va` | Fast voltage-gated Na+ channel | g_Na=120mS/cm², E_Na=50mV |
| **Kv** | `Kv_delayed_rectifier.va` | Delayed rectifier K+ channel | g_K=36mS/cm², E_K=-77mV |
| **CaL** | `CaL_channel.va` | L-type Ca²+ channel | g_Ca=4mS/cm², E_Ca=120mV |
| **KCa** | `KCa_channel.va` | Ca²+-activated K+ channel | g_KCa=5mS/cm², Ca_half=1μM |
| **A-type K+** | `A_type_K.va` | Transient outward K+ channel | g_A=8mS/cm², E_K=-77mV |
| **Leak** | `leak_channel.va` | Passive leak conductance | g_L=0.3mS/cm², E_L=-54.4mV |

### Coupling

| Module | File | Description | Parameters |
|--------|------|-------------|----------|
| **Gap Junction** | `gap_junction.va` | Electrical coupling between cells | g_gap=0.3mS/cm² |

### Pacemakers

| Module | File | Description | Parameters |
|--------|------|-------------|----------|
| **ICC FHN** | `icc_fhn_oscillator.va` | ICC slow wave generator (FHN) | omega=0.005rad/ms, amp=12mV |

---

## Usage Example

### In Cadence Spectre

```spice
* ENS Neuron Circuit
* Load Verilog-A modules
.hdl "NaV1_5.va"
.hdl "Kv_delayed_rectifier.va"
.hdl "CaL_channel.va"
.hdl "leak_channel.va"

* Membrane capacitance
C_mem V_mem 0 1p

* Ion channels
X_na V_mem 0 NaV1_5 g_Na=120m
X_k  V_mem 0 Kv_delayed_rectifier g_K=36m
X_ca V_mem 0 CaL_channel g_Ca=4m
X_l  V_mem 0 leak_channel g_L=0.3m

* Stimulus
I_stim V_mem 0 PULSE(0 10u 10m 1m 1m 50m 100m)

* Simulation
.tran 0.05m 200m
.probe tran v(V_mem)
.end
```

### In HSPICE

```spice
* Include Verilog-A modules
.hdl "verilog_a_library/*.va"

* Instantiate
X_neuron V_mem 0 ens_neuron_cell
.subckt ens_neuron_cell v gnd
    C1 v gnd 1p
    X_na v gnd NaV1_5
    X_k v gnd Kv_delayed_rectifier
    X_ca v gnd CaL_channel
    X_l v gnd leak_channel
.ends
```

---

## Module Details

### NaV1.5 (Fast Sodium Channel)

**Hodgkin-Huxley Na+ channel with m³h gating**

Ports:
- `vp`, `vn`: Positive/negative terminals

Parameters:
- `g_Na`: Maximum conductance (default: 120 mS/cm²)
- `E_Na`: Reversal potential (default: 50 mV)
- `temperature`: Operating temperature (default: 310 K)
- `Q10`: Temperature coefficient (default: 3.0)

Equations:
```
I_Na = g_Na * m³ * h * (V - E_Na)
dm/dt = (m_∞ - m) / τ_m
dh/dt = (h_∞ - h) / τ_h
```

### Kv (Delayed Rectifier K+)

**Hodgkin-Huxley K+ channel with n⁴ gating**

I_K = g_K * n⁴ * (V - E_K)

### CaL (L-type Calcium)

**Voltage-gated Ca²+ channel**

I_Ca = g_Ca * m_Ca * (V - E_Ca)

### KCa (Ca²+-activated K+)

**Calcium-dependent K+ channel (SK-type)**

Requires connection to Ca²+ concentration node.

I_KCa = g_KCa * (Ca²+ⁿ / (Ca²+ⁿ + K_dⁿ)) * (V - E_K)

### Gap Junction

**Electrical coupling between adjacent cells**

I_gap = g_gap * (V₂ - V₁)

Bidirectional, symmetric coupling.

### ICC FHN Oscillator

**Autonomous slow wave generator**

Based on FitzHugh-Nagumo reduced oscillator:
```
dv/dt = v - v³/3 - w
dw/dt = ε(v + a - bw)
```

Generates ~3 cycles/minute slow waves when omega=0.000314 rad/ms.

---

## Testing

### Voltage Clamp Test

```spice
* Test Na+ channel with voltage clamp
V_clamp V_mem 0 PWL(0 -80m 10m -80m 11m 0m 50m 0m 51m -80m)
X_na V_mem 0 NaV1_5
.tran 0.01m 100m
.print tran i(X_na)
```

### Current Clamp Test

```spice
* Test neuron with current injection
I_stim V_mem 0 PULSE(0 10u 10m 1m 1m 20m 100m)
C_mem V_mem 0 1p
X_na V_mem 0 NaV1_5
X_k V_mem 0 Kv_delayed_rectifier
X_l V_mem 0 leak_channel
.tran 0.05m 100m
```

---

## Parameter Mappings

| Python Parameter | Verilog-A Parameter | Units | Notes |
|-----------------|-------------------|-------|-------|
| `g_Na` | `g_Na` | S/cm² | Convert: mS/cm² × 1e-3 |
| `E_Na` | `E_Na` | V | Convert: mV × 1e-3 |
| `C_m` | External capacitor | F/cm² | Add as separate component |

---

## Compilation

### Cadence Spectre

```bash
spectre -64 netlist.spi
```

### HSPICE

```bash
hspice -i netlist.sp -o output
```

### ngspice (limited Verilog-A support)

ngspice has limited Verilog-A support. For ngspice, use SPICE-native subcircuit implementations instead.

---

## References

- Hodgkin & Huxley (1952) *J Physiol* - Original HH model
- FitzHugh (1961) *Biophys J* - Simplified oscillator
- Connor & Stevens (1971) *J Physiol* - A-type K+ channel
- Corrias & Buist (2007) *Biophys J* - ICC calcium clock

---

## Contributing

To add new modules:
1. Follow naming convention: `<ChannelType>_<subtype>.va`
2. Include header comment with reference
3. Use standard port names (`vp`, `vn`, `gnd`)
4. Add to this README
5. Test in at least one simulator

---

## License

MIT License - See main project LICENSE file

---

**Status**: ✅ Standard cell library 90% complete
**Last Updated**: 2026-02-14
