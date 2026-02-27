"""
Scientific validation checks for ENS-GI Digital Twin layers.

This script runs profile simulations and verifies objective biomarker
relationships for:
- Layer 1 (cellular electrophysiology)
- Layer 2 (network propagation)
- Layer 3 (ICC pacing and motility)
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ens_gi_digital import ENSGIDigitalTwin  # noqa: E402


def simulate_profile(profile: str, duration_ms: float, n_segments: int = 20):
    twin = ENSGIDigitalTwin(n_segments=n_segments)
    twin.apply_profile(profile)
    twin.run(
        duration=duration_ms,
        dt=0.05,
        I_stim={3: 12.0, 4: 10.0},
        record=True,
        verbose=False,
    )
    return twin.extract_biomarkers()


def in_range(value: float, low: float, high: float) -> bool:
    return (value >= low) and (value <= high)


def run_checks(duration_ms: float, n_segments: int):
    healthy = simulate_profile("healthy", duration_ms, n_segments)
    ibs_d = simulate_profile("ibs_d", duration_ms, n_segments)
    ibs_c = simulate_profile("ibs_c", duration_ms, n_segments)

    checks = []

    # Layer 1 checks
    checks.append(
        (
            "L1 mean_membrane_potential plausible",
            in_range(healthy["mean_membrane_potential"], -90.0, 20.0),
            healthy["mean_membrane_potential"],
        )
    )
    checks.append(
        (
            "L1 voltage_variance > 0",
            healthy["voltage_variance"] > 0.0,
            healthy["voltage_variance"],
        )
    )
    checks.append(
        (
            "L1 calcium metrics are physically consistent",
            healthy["mean_calcium"] > 0.0 and healthy["peak_calcium"] >= healthy["mean_calcium"],
            (healthy["mean_calcium"], healthy["peak_calcium"]),
        )
    )

    # Layer 2 checks
    checks.append(
        (
            "L2 propagation_correlation in [-1, 1]",
            in_range(healthy["propagation_correlation"], -1.0, 1.0),
            healthy["propagation_correlation"],
        )
    )
    checks.append(
        (
            "L2 ICC propagation uniformity in [0, 1]",
            in_range(healthy["icc_propagation"]["propagation_uniformity"], 0.0, 1.0),
            healthy["icc_propagation"]["propagation_uniformity"],
        )
    )

    # Layer 3 checks
    checks.append(
        (
            "L3 healthy ICC frequency in [2, 4] cpm",
            in_range(healthy["icc_frequency_cpm"], 2.0, 4.0),
            healthy["icc_frequency_cpm"],
        )
    )
    checks.append(
        (
            "L3 motility_index in [0, 100]",
            in_range(healthy["motility_index"], 0.0, 100.0),
            healthy["motility_index"],
        )
    )
    checks.append(
        (
            "L3 profile ordering: IBS-D ICC > Healthy > IBS-C ICC",
            ibs_d["icc_frequency_cpm"] > healthy["icc_frequency_cpm"] > ibs_c["icc_frequency_cpm"],
            (
                ibs_d["icc_frequency_cpm"],
                healthy["icc_frequency_cpm"],
                ibs_c["icc_frequency_cpm"],
            ),
        )
    )

    print("=" * 78)
    print("ENS-GI scientific validation")
    print("=" * 78)
    print(f"Duration: {duration_ms:.0f} ms | Segments: {n_segments}")
    print("")

    print("Biomarker snapshot:")
    print(
        "Healthy: ICC={:.3f} cpm, spike_rate={:.3f}, motility={:.3f}, corr={:.3f}".format(
            healthy["icc_frequency_cpm"],
            healthy["spike_rate_per_neuron"],
            healthy["motility_index"],
            healthy["propagation_correlation"],
        )
    )
    print(
        "IBS-D : ICC={:.3f} cpm, spike_rate={:.3f}, motility={:.3f}".format(
            ibs_d["icc_frequency_cpm"],
            ibs_d["spike_rate_per_neuron"],
            ibs_d["motility_index"],
        )
    )
    print(
        "IBS-C : ICC={:.3f} cpm, spike_rate={:.3f}, motility={:.3f}".format(
            ibs_c["icc_frequency_cpm"],
            ibs_c["spike_rate_per_neuron"],
            ibs_c["motility_index"],
        )
    )
    print("")

    failed = 0
    print("Checks:")
    for name, ok, value in checks:
        if isinstance(value, tuple):
            value_text = ", ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in value)
        elif isinstance(value, float):
            if math.isfinite(value):
                value_text = f"{value:.4f}"
            else:
                value_text = str(value)
        else:
            value_text = str(value)

        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name} | value={value_text}")
        if not ok:
            failed += 1

    print("")
    if failed == 0:
        print("Result: PASS (all scientific checks satisfied)")
        return 0

    print(f"Result: FAIL ({failed} checks failed)")
    return 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-ms", type=float, default=2000.0)
    parser.add_argument("--segments", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(run_checks(args.duration_ms, args.segments))
