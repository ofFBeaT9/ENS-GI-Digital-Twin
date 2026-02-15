#!/usr/bin/env python3
"""
Validates SPICE netlist by running in ngspice.

Tests both pure SPICE and Verilog-A netlists (though Verilog-A
may not work in ngspice).
"""
import subprocess
import os
from pathlib import Path
import sys

# Configure UTF-8 output for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

NGSPICE_PATH = r'c:\ens-gi digital\Spice64\bin\ngspice.exe'

def test_pure_spice_netlist():
    """Generate and test pure SPICE netlist."""
    from ens_gi_digital import ENSGIDigitalTwin

    print("\n" + "="*70)
    print("TEST 1: Pure SPICE Netlist (ngspice compatible)")
    print("="*70)

    twin = ENSGIDigitalTwin(n_segments=5)  # Small for testing
    twin.apply_profile('healthy')

    # Generate pure SPICE netlist
    netlist_path = 'test_network_pure.sp'
    twin.export_spice_netlist(netlist_path, use_verilog_a=False)
    print(f"✓ Generated pure SPICE netlist: {netlist_path}")

    # Check that ngspice executable exists
    if not os.path.exists(NGSPICE_PATH):
        print(f"✗ ngspice not found at {NGSPICE_PATH}")
        return False

    # Run in ngspice batch mode
    print(f"\nRunning ngspice simulation...")
    try:
        result = subprocess.run(
            [NGSPICE_PATH, '-b', '-r', 'test_output.raw', '-o', 'test_output.log', netlist_path],
            capture_output=True,
            text=True,
            timeout=60
        )

        # Check for errors in output
        if result.returncode == 0:
            print("✓ Pure SPICE netlist runs successfully (exit code 0)")

            # Check if output file was generated
            if os.path.exists('test_output.raw'):
                print("✓ Output file generated: test_output.raw")

            # Show any warnings
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                warnings = [l for l in lines if 'warning' in l.lower()]
                if warnings:
                    print(f"\n⚠ {len(warnings)} warnings found:")
                    for w in warnings[:5]:  # Show first 5
                        print(f"  {w}")

            return True
        else:
            print(f"✗ Pure SPICE netlist failed (exit code {result.returncode})")
            print("\nSTDERR:")
            print(result.stderr)
            print("\nSTDOUT:")
            print(result.stdout)
            return False

    except subprocess.TimeoutExpired:
        print("✗ Simulation timed out (>60s)")
        return False
    except Exception as e:
        print(f"✗ Error running ngspice: {e}")
        return False


def test_verilog_a_netlist():
    """Test Verilog-A netlist (may not work in ngspice)."""
    from ens_gi_digital import ENSGIDigitalTwin

    print("\n" + "="*70)
    print("TEST 2: Verilog-A Netlist (may require Spectre/HSPICE)")
    print("="*70)

    twin = ENSGIDigitalTwin(n_segments=5)
    twin.apply_profile('healthy')

    netlist_path = 'test_network_va.sp'
    twin.export_spice_netlist(netlist_path, use_verilog_a=True)

    print(f"✓ Generated Verilog-A netlist: {netlist_path}")
    print("⚠ Verilog-A netlist generated but may require Spectre/HSPICE/Xyce")
    print("  ngspice has limited Verilog-A support via OSDI/OpenVAF")

    return True


def analyze_netlist_structure(filename='test_network_pure.sp'):
    """Analyze the structure of generated netlist."""
    print("\n" + "="*70)
    print(f"NETLIST ANALYSIS: {filename}")
    print("="*70)

    if not os.path.exists(filename):
        print(f"✗ File not found: {filename}")
        return

    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Count components
    subcircuits = [l for l in lines if l.strip().startswith('.subckt')]
    capacitors = [l for l in lines if l.strip().startswith('C_')]
    current_sources = [l for l in lines if l.strip().startswith('I_')]
    resistors = [l for l in lines if l.strip().startswith('R_')]
    subckt_instances = [l for l in lines if l.strip().startswith('X_')]

    print(f"Total lines: {len(lines)}")
    print(f"Subcircuit definitions: {len(subcircuits)}")
    for sc in subcircuits:
        name = sc.split()[1]
        print(f"  - {name}")

    print(f"\nComponent counts:")
    print(f"  Capacitors (C_*): {len(capacitors)}")
    print(f"  Current sources (I_*): {len(current_sources)}")
    print(f"  Resistors (R_*): {len(resistors)}")
    print(f"  Subcircuit instances (X_*): {len(subckt_instances)}")

    # Check for common errors
    print(f"\nError checks:")
    errors = []

    # Check for undefined subcircuits
    defined_subcircuits = set()
    for sc in subcircuits:
        tokens = sc.split()
        if len(tokens) >= 2:
            defined_subcircuits.add(tokens[1])

    used_subcircuits = set()
    for inst in subckt_instances:
        tokens = inst.strip().split()
        if len(tokens) >= 3:
            # Format: X_name node1 node2 subcircuit_name
            subckt_name = tokens[-1]
            used_subcircuits.add(subckt_name)

    undefined = used_subcircuits - defined_subcircuits
    if undefined:
        errors.append(f"Undefined subcircuits: {undefined}")

    # Check for old syntax issues
    for i, line in enumerate(lines):
        if "cur='" in line:
            errors.append(f"Line {i+1}: Old behavioral source syntax 'cur=' found")
        if '^' in line and '.subckt' not in line:
            # Check if it's in an expression context
            if 'VALUE=' in line or 'cur=' in line:
                errors.append(f"Line {i+1}: Exponentiation operator '^' (should use pow() or **)")

    if errors:
        print("  ✗ Errors found:")
        for err in errors:
            print(f"    - {err}")
    else:
        print("  ✓ No obvious errors detected")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("ENS-GI DIGITAL TWIN — SPICE NETLIST VALIDATION")
    print("="*70)

    success = True

    # Test 1: Pure SPICE
    try:
        if not test_pure_spice_netlist():
            success = False
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        success = False

    # Analyze structure
    try:
        analyze_netlist_structure('test_network_pure.sp')
    except Exception as e:
        print(f"✗ Analysis failed: {e}")

    # Test 2: Verilog-A (informational only)
    try:
        test_verilog_a_netlist()
    except Exception as e:
        print(f"✗ Verilog-A test failed: {e}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    if success:
        print("✓ All critical tests passed")
        print("  Pure SPICE netlist is ready for ngspice")
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        print("  Check errors above and fix netlist generation")
        sys.exit(1)
