"""
Pre-flight validation for SPICE/Verilog-A export with biologically plausible parameters.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ens_gi_digital.core import ENSGIDigitalTwin  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=int, default=20)
    parser.add_argument("--out-dir", default="preflight_exports")
    parser.add_argument("--run-pytest", action="store_true")
    return parser.parse_args()


def set_dummy_params(twin: ENSGIDigitalTwin):
    for neuron in twin.network.neurons:
        neuron.params.g_Na = 120.0
        neuron.params.g_K = 36.0
        neuron.params.g_Ca = 6.5
    twin.icc.params.omega = 0.00045
    twin.network.params.coupling_strength = 0.15


def validate_text_outputs(spice_text: str, verilog_text: str):
    checks = [
        (".end" in spice_text.lower(), "SPICE has .end"),
        ("module" in verilog_text.lower(), "Verilog-A has module"),
        ("endmodule" in verilog_text.lower(), "Verilog-A has endmodule"),
        ("g_na" in spice_text.lower(), "SPICE includes g_Na parameter"),
        ("g_k" in spice_text.lower(), "SPICE includes g_K parameter"),
        ("g_ca" in spice_text.lower(), "SPICE includes g_Ca parameter"),
    ]
    failed = [name for ok, name in checks if not ok]
    if failed:
        raise RuntimeError("Export validation failed: " + ", ".join(failed))


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    twin = ENSGIDigitalTwin(n_segments=args.segments)
    set_dummy_params(twin)

    spice_path = out_dir / "preflight_netlist.sp"
    spice_va_path = out_dir / "preflight_netlist_va.sp"
    verilog_path = out_dir / "preflight_ens_neuron.va"

    spice_text = twin.export_spice_netlist(filename=str(spice_path), use_verilog_a=False)
    _ = twin.export_spice_netlist(filename=str(spice_va_path), use_verilog_a=True)
    verilog_text = twin.export_verilog_a_module()
    verilog_path.write_text(verilog_text, encoding="utf-8")

    validate_text_outputs(spice_text, verilog_text)

    print("[OK] Export preflight passed")
    print(f"[OK] Wrote: {spice_path}")
    print(f"[OK] Wrote: {spice_va_path}")
    print(f"[OK] Wrote: {verilog_path}")

    if args.run_pytest:
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_hardware_export.py",
            "-k",
            "SPICENetlistGeneration or VerilogAExport",
            "-q",
        ]
        print("[INFO] Running:", " ".join(cmd))
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            return proc.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
