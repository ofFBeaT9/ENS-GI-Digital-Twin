"""
SPICE Netlist Validation Script for ENS-GI Digital Twin

This script:
1. Exports SPICE netlist from the digital twin
2. Executes ngspice simulation
3. Parses SPICE output voltages
4. Runs equivalent Python simulation
5. Compares results and generates validation report

Usage:
    python validate_spice.py

Requirements:
    - ngspice installed at: c:\ens-gi digital\Spice64\bin\ngspice.exe
    - ENS-GI Digital Twin installed (ens_gi_core.py in path)
"""

import numpy as np
import subprocess
import re
import os
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from dataclasses import dataclass

from ens_gi_digital import ENSGIDigitalTwin


@dataclass
class ValidationResult:
    """Results from SPICE validation run."""
    success: bool
    correlation: float
    frequency_match: bool
    propagation_match: bool
    errors: List[str]
    warnings: List[str]

    def __str__(self) -> str:
        status = "✅ PASS" if self.success else "❌ FAIL"
        lines = [
            f"\n{'='*60}",
            f"SPICE VALIDATION REPORT - {status}",
            f"{'='*60}",
            f"Voltage Correlation: {self.correlation:.3f} (target: >0.95)",
            f"Frequency Match: {'✓' if self.frequency_match else '✗'}",
            f"Propagation Match: {'✓' if self.propagation_match else '✗'}",
        ]

        if self.errors:
            lines.append(f"\n🔴 ERRORS ({len(self.errors)}):")
            for err in self.errors:
                lines.append(f"  - {err}")

        if self.warnings:
            lines.append(f"\n⚠️ WARNINGS ({len(self.warnings)}):")
            for warn in self.warnings:
                lines.append(f"  - {warn}")

        lines.append("="*60 + "\n")
        return "\n".join(lines)


class SPICEValidator:
    """Validates SPICE netlists against Python simulation."""

    def __init__(self, ngspice_path: str = r"c:\ens-gi digital\Spice64\bin\ngspice.exe"):
        """Initialize validator.

        Args:
            ngspice_path: Path to ngspice executable
        """
        self.ngspice_path = Path(ngspice_path)
        self.netlist_file = Path("test_network.sp")
        self.output_file = Path("ngspice_output.txt")

        # Validation thresholds
        self.min_correlation = 0.95
        self.freq_tolerance = 0.1  # 10% tolerance on frequency

    def validate(self, n_segments: int = 5, profile: str = "healthy") -> ValidationResult:
        """Run full validation workflow.

        Args:
            n_segments: Number of network segments to test
            profile: IBS profile to test

        Returns:
            ValidationResult with pass/fail status and metrics
        """
        print(f"\n{'='*60}")
        print(f"Starting SPICE Validation")
        print(f"Profile: {profile}, Segments: {n_segments}")
        print(f"{'='*60}\n")

        errors = []
        warnings = []

        # Step 1: Check ngspice exists
        if not self.ngspice_path.exists():
            errors.append(f"ngspice not found at: {self.ngspice_path}")
            return ValidationResult(False, 0.0, False, False, errors, warnings)

        # Step 2: Run Python simulation
        print("Step 1/5: Running Python simulation...")
        try:
            python_time, python_voltages = self._run_python_simulation(n_segments, profile)
            print(f"  ✓ Python simulation complete ({len(python_time)} timesteps)")
        except Exception as e:
            errors.append(f"Python simulation failed: {str(e)}")
            return ValidationResult(False, 0.0, False, False, errors, warnings)

        # Step 3: Export SPICE netlist
        print("\nStep 2/5: Exporting SPICE netlist...")
        try:
            self._export_spice_netlist(n_segments, profile)
            print(f"  ✓ Netlist exported to: {self.netlist_file}")
        except Exception as e:
            errors.append(f"SPICE export failed: {str(e)}")
            return ValidationResult(False, 0.0, False, False, errors, warnings)

        # Step 4: Run ngspice
        print("\nStep 3/5: Running ngspice simulation...")
        try:
            spice_success = self._run_ngspice()
            if not spice_success:
                errors.append("ngspice simulation failed (see output)")
                return ValidationResult(False, 0.0, False, False, errors, warnings)
            print(f"  ✓ ngspice simulation complete")
        except Exception as e:
            errors.append(f"ngspice execution failed: {str(e)}")
            return ValidationResult(False, 0.0, False, False, errors, warnings)

        # Step 5: Parse SPICE output
        print("\nStep 4/5: Parsing SPICE output...")
        try:
            spice_time, spice_voltages = self._parse_spice_output()
            print(f"  ✓ Parsed {len(spice_time)} timesteps")
        except Exception as e:
            errors.append(f"SPICE output parsing failed: {str(e)}")
            return ValidationResult(False, 0.0, False, False, errors, warnings)

        # Step 6: Compare results
        print("\nStep 5/5: Comparing results...")
        try:
            comparison = self._compare_results(
                python_time, python_voltages,
                spice_time, spice_voltages,
                n_segments
            )
            print(f"  ✓ Comparison complete")
        except Exception as e:
            errors.append(f"Result comparison failed: {str(e)}")
            return ValidationResult(False, 0.0, False, False, errors, warnings)

        # Generate plots
        print("\n  Generating comparison plots...")
        self._generate_plots(
            python_time, python_voltages,
            spice_time, spice_voltages,
            n_segments
        )
        print(f"  ✓ Plots saved to: validation_plot.png")

        # Evaluate results
        correlation = comparison['correlation']
        freq_match = comparison['freq_match']
        prop_match = comparison['prop_match']

        # Check pass criteria
        success = (
            correlation >= self.min_correlation and
            freq_match and
            prop_match
        )

        if correlation < self.min_correlation:
            errors.append(f"Voltage correlation {correlation:.3f} < {self.min_correlation}")
        if not freq_match:
            warnings.append("Oscillation frequency mismatch")
        if not prop_match:
            warnings.append("Propagation delay mismatch")

        return ValidationResult(success, correlation, freq_match, prop_match, errors, warnings)

    def _run_python_simulation(self, n_segments: int, profile: str) -> Tuple[np.ndarray, np.ndarray]:
        """Run Python digital twin simulation.

        Returns:
            (time, voltages) where voltages is shape (n_timesteps, n_segments)
        """
        dt = ENSGIDigitalTwin(n_segments=n_segments, profile=profile)

        # Run simulation for 1000 ms
        duration_ms = 1000
        n_steps = int(duration_ms / dt.params.dt)

        time = []
        voltages = []

        for _ in range(n_steps):
            time.append(dt.t)
            V = [neuron.V for neuron in dt.network.neurons]
            voltages.append(V)
            dt.step()

        return np.array(time), np.array(voltages)

    def _export_spice_netlist(self, n_segments: int, profile: str):
        """Export SPICE netlist to file."""
        dt = ENSGIDigitalTwin(n_segments=n_segments, profile=profile)
        dt.export_spice_netlist(filename=str(self.netlist_file), use_verilog_a=False)

    def _run_ngspice(self) -> bool:
        """Execute ngspice on the netlist.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Run ngspice in batch mode
            cmd = [str(self.ngspice_path), "-b", str(self.netlist_file), "-o", str(self.output_file)]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout
            )

            # Check for errors in output
            if result.returncode != 0:
                print(f"  ✗ ngspice returned error code: {result.returncode}")
                print(f"  stderr: {result.stderr}")
                return False

            return True

        except subprocess.TimeoutExpired:
            print("  ✗ ngspice simulation timed out (>60s)")
            return False
        except Exception as e:
            print(f"  ✗ ngspice execution error: {str(e)}")
            return False

    def _parse_spice_output(self) -> Tuple[np.ndarray, np.ndarray]:
        """Parse ngspice output file.

        Returns:
            (time, voltages) arrays
        """
        if not self.output_file.exists():
            raise FileNotFoundError(f"ngspice output not found: {self.output_file}")

        with open(self.output_file, 'r') as f:
            content = f.read()

        # Find the data section
        # ngspice output format: Index  time  v(v0)  v(v1)  v(v2) ...
        lines = content.split('\n')

        time_data = []
        voltage_data = []

        in_data_section = False
        for line in lines:
            # Look for data rows (start with index number)
            if re.match(r'^\s*\d+\s+', line):
                in_data_section = True
                parts = line.split()
                if len(parts) >= 3:  # index, time, at least one voltage
                    try:
                        t = float(parts[1])
                        v_values = [float(parts[i]) for i in range(2, len(parts))]
                        time_data.append(t)
                        voltage_data.append(v_values)
                    except ValueError:
                        continue

        if not time_data:
            raise ValueError("No data found in ngspice output")

        return np.array(time_data), np.array(voltage_data)

    def _compare_results(
        self,
        python_time: np.ndarray,
        python_voltages: np.ndarray,
        spice_time: np.ndarray,
        spice_voltages: np.ndarray,
        n_segments: int
    ) -> Dict:
        """Compare Python and SPICE simulation results.

        Returns:
            Dictionary with comparison metrics
        """
        # Interpolate SPICE to Python time grid (if needed)
        if len(spice_time) != len(python_time):
            # Simple resampling - take every nth point or interpolate
            from scipy.interpolate import interp1d
            spice_interp = interp1d(spice_time, spice_voltages, axis=0, fill_value='extrapolate')
            spice_voltages_resampled = spice_interp(python_time)
        else:
            spice_voltages_resampled = spice_voltages

        # Compute correlation for first neuron
        v0_python = python_voltages[:, 0]
        v0_spice = spice_voltages_resampled[:, 0]
        correlation = np.corrcoef(v0_python, v0_spice)[0, 1]

        # Analyze oscillation frequency (using FFT)
        python_freq = self._estimate_frequency(python_time, v0_python)
        spice_freq = self._estimate_frequency(python_time, v0_spice)

        freq_match = abs(python_freq - spice_freq) / python_freq < self.freq_tolerance

        # Check propagation delay (if multi-segment)
        prop_match = True
        if n_segments > 1:
            # Measure time to peak for first vs last neuron
            python_delay = self._measure_propagation_delay(python_time, python_voltages)
            spice_delay = self._measure_propagation_delay(python_time, spice_voltages_resampled)

            if python_delay > 0 and spice_delay > 0:
                prop_match = abs(python_delay - spice_delay) / python_delay < 0.2  # 20% tolerance

        return {
            'correlation': correlation,
            'freq_match': freq_match,
            'prop_match': prop_match,
            'python_freq': python_freq,
            'spice_freq': spice_freq,
        }

    def _estimate_frequency(self, time: np.ndarray, voltage: np.ndarray) -> float:
        """Estimate dominant frequency using FFT.

        Returns:
            Frequency in cycles per minute
        """
        # Remove DC component
        voltage_ac = voltage - np.mean(voltage)

        # Compute FFT
        dt = time[1] - time[0] if len(time) > 1 else 1.0
        fft = np.fft.rfft(voltage_ac)
        freqs = np.fft.rfftfreq(len(voltage_ac), dt / 1000)  # Convert to Hz

        # Find peak frequency (ignore DC)
        power = np.abs(fft[1:])
        peak_idx = np.argmax(power) + 1
        peak_freq_hz = freqs[peak_idx]

        # Convert to cycles per minute
        return peak_freq_hz * 60

    def _measure_propagation_delay(self, time: np.ndarray, voltages: np.ndarray) -> float:
        """Measure propagation delay from first to last neuron.

        Returns:
            Delay in milliseconds
        """
        # Find time of first peak in first neuron
        v_first = voltages[:, 0]
        v_last = voltages[:, -1]

        # Find peaks
        from scipy.signal import find_peaks
        peaks_first, _ = find_peaks(v_first, height=-40, distance=50)
        peaks_last, _ = find_peaks(v_last, height=-40, distance=50)

        if len(peaks_first) > 0 and len(peaks_last) > 0:
            t_first = time[peaks_first[0]]
            t_last = time[peaks_last[0]]
            return t_last - t_first

        return 0.0

    def _generate_plots(
        self,
        python_time: np.ndarray,
        python_voltages: np.ndarray,
        spice_time: np.ndarray,
        spice_voltages: np.ndarray,
        n_segments: int
    ):
        """Generate comparison plots."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Plot voltage traces
        ax = axes[0]
        ax.plot(python_time, python_voltages[:, 0], 'b-', label='Python V0', linewidth=2)
        ax.plot(spice_time, spice_voltages[:, 0], 'r--', label='SPICE V0', linewidth=1.5, alpha=0.7)

        if n_segments > 1:
            ax.plot(python_time, python_voltages[:, -1], 'b-', label=f'Python V{n_segments-1}', alpha=0.5)
            ax.plot(spice_time, spice_voltages[:, -1], 'r--', label=f'SPICE V{n_segments-1}', alpha=0.5)

        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Voltage (mV)')
        ax.set_title('Python vs SPICE Voltage Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot error
        ax = axes[1]
        if len(spice_time) == len(python_time):
            error = python_voltages[:, 0] - spice_voltages[:, 0]
            ax.plot(python_time, error, 'k-', linewidth=1)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Error (mV)')
            ax.set_title('Voltage Error (Python - SPICE)')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Time grids differ - interpolation needed',
                   ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()
        plt.savefig('validation_plot.png', dpi=150)
        print(f"    Saved: validation_plot.png")


def main():
    """Run SPICE validation."""
    validator = SPICEValidator()

    # Test healthy 5-segment network
    result = validator.validate(n_segments=5, profile="healthy")

    # Print results
    print(result)

    # Return exit code
    return 0 if result.success else 1


if __name__ == "__main__":
    exit(main())
