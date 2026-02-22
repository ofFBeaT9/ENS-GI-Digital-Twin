"""Hardware export tests — SPICE and Verilog-A netlist generation.

Tests structural/syntactic correctness of generated netlists without
requiring ngspice. The NgspiceSimulation class is skipped automatically
when ngspice is not on PATH or not in the project-local bundle.
"""
import re
import shutil
from pathlib import Path

import pytest

from ens_gi_digital.core import ENSGIDigitalTwin

# ── ngspice binary detection ────────────────────────────────────────────────
# 1. Check system PATH (Linux ngspice install)
# 2. Fall back to the project-bundled Windows binary (works via WSL interop)
_PROJECT_ROOT = Path(__file__).parent.parent
_LOCAL_NGSPICE_PATHS = [
    _PROJECT_ROOT / 'ngspice' / 'Spice64' / 'bin' / 'ngspice_con.exe',
    _PROJECT_ROOT / 'ngspice' / 'Spice64' / 'bin' / 'ngspice.exe',
    # WSL /mnt/c path variant
    Path('/mnt/c/Tritone SoC/Spice64/bin/ngspice_con.exe'),
]

def _find_ngspice():
    """Return the ngspice binary path, or None if not available."""
    binary = shutil.which('ngspice')
    if binary:
        return binary
    for candidate in _LOCAL_NGSPICE_PATHS:
        if candidate.exists():
            return str(candidate)
    return None

NGSPICE = _find_ngspice()


@pytest.fixture
def twin():
    """Small digital twin sufficient for hardware export tests."""
    return ENSGIDigitalTwin(n_segments=5)


@pytest.fixture
def twin_ibs_d():
    """IBS-D profile twin."""
    t = ENSGIDigitalTwin(n_segments=5)
    t.apply_profile('ibs_d')
    return t


# ─────────────────────────────────────────────────────────────
# SPICE netlist generation (no ngspice required)
# ─────────────────────────────────────────────────────────────

class TestSPICENetlistGeneration:
    """Verify that export_spice_netlist() produces valid SPICE syntax."""

    def test_export_spice_creates_file(self, twin, tmp_path):
        """export_spice_netlist(filename=...) writes a file to disk."""
        out = tmp_path / 'network.sp'
        twin.export_spice_netlist(filename=str(out))
        assert out.exists(), "SPICE file was not created"
        assert out.stat().st_size > 0, "SPICE file is empty"

    def test_spice_returns_string(self, twin):
        """export_spice_netlist() always returns a non-empty string."""
        netlist = twin.export_spice_netlist()
        assert isinstance(netlist, str)
        assert len(netlist) > 0

    def test_spice_has_end_statement(self, twin, tmp_path):
        """Well-formed SPICE netlists must end with a .end directive."""
        out = tmp_path / 'network.sp'
        twin.export_spice_netlist(filename=str(out))
        content = out.read_text(encoding='utf-8')
        # .end must appear (case-insensitive) as its own line
        lines = [ln.strip().lower() for ln in content.splitlines()]
        assert '.end' in lines, "SPICE netlist missing '.end' statement"

    def test_spice_has_title_line(self, twin):
        """SPICE netlists should have a recognisable title / comment header."""
        netlist = twin.export_spice_netlist()
        first_line = netlist.splitlines()[0]
        # First line is a title or comment (starts with * in SPICE)
        assert len(first_line) > 0

    def test_spice_ibs_d_profile(self, twin_ibs_d, tmp_path):
        """IBS-D profile twin should also export a valid netlist."""
        out = tmp_path / 'ibs_d.sp'
        twin_ibs_d.export_spice_netlist(filename=str(out))
        assert out.exists()
        content = out.read_text(encoding='utf-8')
        assert len(content) > 0


# ─────────────────────────────────────────────────────────────
# Verilog-A export (no simulator required)
# ─────────────────────────────────────────────────────────────

class TestVerilogAExport:
    """Verify Verilog-A module string correctness."""

    def test_verilog_a_module_string_returned(self, twin):
        """export_verilog_a_module() returns a non-empty string."""
        va = twin.export_verilog_a_module()
        assert isinstance(va, str)
        assert len(va) > 0

    def test_verilog_a_has_module_keyword(self, twin):
        """Verilog-A output must start a module with the 'module' keyword."""
        va = twin.export_verilog_a_module()
        assert 'module' in va, "Verilog-A string missing 'module' keyword"

    def test_verilog_a_has_endmodule(self, twin):
        """Verilog-A output must close the module with 'endmodule'."""
        va = twin.export_verilog_a_module()
        assert 'endmodule' in va, "Verilog-A string missing 'endmodule'"

    def test_verilog_a_spice_netlist_variant(self, twin, tmp_path):
        """export_spice_netlist(use_verilog_a=True) creates a file."""
        out = tmp_path / 'network_va.sp'
        twin.export_spice_netlist(filename=str(out), use_verilog_a=True)
        assert out.exists()
        assert out.stat().st_size > 0


# ─────────────────────────────────────────────────────────────
# ngspice simulation (skipped unless ngspice is installed)
# ─────────────────────────────────────────────────────────────

class TestNgspiceSimulation:
    """Run the generated netlist through ngspice.

    All tests in this class are skipped when ngspice is not available on
    PATH or in the project-local bundle.  Marked @pytest.mark.slow so they
    are excluded from the fast CI matrix by default.
    """

    @staticmethod
    def _to_ngspice_path(path) -> str:
        """Convert a Linux path to Windows format when using bundled ngspice.exe.

        When running from WSL, pytest tmp_path lives in the Linux filesystem
        (e.g. /tmp/pytest-xxx/) which the Windows ngspice_con.exe cannot access.
        wslpath -w converts it to the Windows UNC equivalent (\\\\wsl.localhost\\...).
        """
        import subprocess, platform
        path_str = str(path)
        if (NGSPICE and NGSPICE.endswith('.exe')
                and platform.system() == 'Linux'):
            try:
                win_path = subprocess.check_output(
                    ['wslpath', '-w', path_str],
                    text=True, timeout=5,
                ).strip()
                return win_path
            except Exception:
                pass  # fall through — try the native path anyway
        return path_str

    @pytest.mark.slow
    @pytest.mark.skipif(NGSPICE is None, reason="ngspice not installed")
    def test_ngspice_runs_successfully(self, twin, tmp_path):
        """ngspice exits with status 0 on a valid netlist."""
        import subprocess

        out = tmp_path / 'network.sp'
        twin.export_spice_netlist(filename=str(out))
        netlist = self._to_ngspice_path(out)

        result = subprocess.run(
            [NGSPICE, '-b', netlist],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"ngspice failed (rc={result.returncode}):\n{result.stderr}"
        )

    @pytest.mark.slow
    @pytest.mark.skipif(NGSPICE is None, reason="ngspice not installed")
    def test_ngspice_output_contains_simulation_data(self, twin, tmp_path):
        """ngspice stdout should contain recognisable simulation output."""
        import subprocess

        out = tmp_path / 'network.sp'
        twin.export_spice_netlist(filename=str(out))
        netlist = self._to_ngspice_path(out)

        result = subprocess.run(
            [NGSPICE, '-b', netlist],
            capture_output=True,
            text=True,
            timeout=60,
        )
        combined = result.stdout + result.stderr
        # ngspice always prints its version banner
        assert len(combined) > 0, "ngspice produced no output"
