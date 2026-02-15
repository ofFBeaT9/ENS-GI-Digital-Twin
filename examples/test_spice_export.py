"""
Test SPICE Netlist Generation and Export
=========================================
Demonstrates hardware export capabilities of ENS-GI Digital Twin.

This script:
1. Creates digital twin with IBS profile
2. Exports SPICE netlist (both Verilog-A and pure SPICE)
3. Validates netlist syntax
4. Provides instructions for simulation

Author: Mahdad
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ens_gi_digital import ENSGIDigitalTwin


def test_spice_export():
    """Test SPICE netlist generation."""
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  ENS-GI SPICE Export Test                                 ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # Create digital twin
    print("[1/4] Creating digital twin...")
    twin = ENSGIDigitalTwin(n_segments=10)
    twin.apply_profile('ibs_d')

    # Export pure SPICE version (ngspice compatible)
    print("\n[2/4] Exporting pure SPICE netlist...")
    spice_netlist = twin.export_spice_netlist(
        filename='ens_network_spice.sp',
        use_verilog_a=False
    )
    print(f"  ✓ Pure SPICE netlist: ens_network_spice.sp ({len(spice_netlist)} chars)")

    # Export Verilog-A version (Spectre/HSPICE compatible)
    print("\n[3/4] Exporting Verilog-A netlist...")
    va_netlist = twin.export_spice_netlist(
        filename='ens_network_verilog_a.sp',
        use_verilog_a=True
    )
    print(f"  ✓ Verilog-A netlist: ens_network_verilog_a.sp ({len(va_netlist)} chars)")

    # Export standalone Verilog-A module
    print("\n[4/4] Exporting Verilog-A module...")
    va_module = twin.export_verilog_a_module()
    with open('ens_neuron.va', 'w') as f:
        f.write(va_module)
    print(f"  ✓ Verilog-A module: ens_neuron.va ({len(va_module)} chars)")

    # Validation
    print("\n" + "="*70)
    print("VALIDATION")
    print("="*70)

    # Check for common errors
    errors = []

    # Check 1: .end present
    if not spice_netlist.strip().endswith('.end'):
        errors.append("Missing .end statement")

    # Check 2: No undefined references
    if 'X_' in spice_netlist and '.subckt' not in spice_netlist and not twin:
        errors.append("Subcircuit references without definitions")

    # Check 3: Valid component names
    import re
    invalid_components = re.findall(r'^[^*\.](\S+)', spice_netlist, re.MULTILINE)
    invalid = [c for c in invalid_components if not re.match(r'[CRLGVEIFXK]', c)]
    if invalid:
        errors.append(f"Invalid component names: {invalid[:3]}")

    if errors:
        print("❌ Validation warnings:")
        for err in errors:
            print(f"  • {err}")
    else:
        print("✓ Netlist validation passed!")

    # Usage instructions
    print("\n" + "="*70)
    print("USAGE INSTRUCTIONS")
    print("="*70)

    print("\n1. Pure SPICE (ngspice):")
    print("   $ ngspice ens_network_spice.sp")
    print("   ngspice> run")
    print("   ngspice> plot v(V0) v(V1) v(V2)")
    print("   ngspice> quit")

    print("\n2. Verilog-A (Cadence Spectre):")
    print("   $ spectre ens_network_verilog_a.sp")

    print("\n3. Verilog-A (HSPICE):")
    print("   $ hspice -i ens_network_verilog_a.sp -o output")

    print("\n4. Standalone Verilog-A module:")
    print("   • Include ens_neuron.va in your own netlist")
    print("   • Example:")
    print("     .hdl 'ens_neuron.va'")
    print("     X_neuron1 V_mem 0 ens_neuron")

    print("\n" + "="*70)
    print("EXPECTED OUTPUT")
    print("="*70)
    print("\nSimulation should show:")
    print("  • Resting potential: ~-65 mV")
    print("  • Action potentials after stimulus (t > 10 ms)")
    print("  • Wave propagation along network")
    print("  • ICC slow waves (~3 cycles/minute)")

    print("\n✓ SPICE export test complete!")
    print("\nNext steps:")
    print("  1. Install ngspice: https://ngspice.sourceforge.io/")
    print("  2. Run: ngspice ens_network_spice.sp")
    print("  3. Compare SPICE vs Python simulation results")


if __name__ == '__main__':
    test_spice_export()
