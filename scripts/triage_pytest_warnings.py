"""
Run small pytest batches with warnings treated as errors.

Purpose:
- Surface the first warning in each critical module quickly.
- Produce a compact pre-flight warning triage report before long PINN runs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_TARGETS = [
    "tests/test_pinn.py::TestPINNEstimator::test_training_step",
    "tests/test_pinn.py::TestPINNEstimator::test_train_on_small_dataset",
    "tests/test_bayesian.py::TestMCMCSampling::test_mcmc_runs",
    "tests/test_hardware_export.py::TestSPICENetlistGeneration::test_spice_returns_string",
]


def run_target(target: str) -> tuple[int, str]:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        target,
        "-W",
        "error",
        "--maxfail=1",
        "-q",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, output.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", action="append", default=None, help="Optional pytest target (repeatable).")
    parser.add_argument("--save-report", default="warning_triage_report.txt")
    args = parser.parse_args()

    targets = args.target if args.target else DEFAULT_TARGETS
    report_lines = []
    failures = 0

    report_lines.append("=" * 78)
    report_lines.append("Pytest Warning Triage (-W error)")
    report_lines.append("=" * 78)

    for target in targets:
        rc, output = run_target(target)
        status = "PASS" if rc == 0 else "FAIL"
        if rc != 0:
            failures += 1
        report_lines.append("")
        report_lines.append(f"[{status}] {target}")
        report_lines.append(output[-4000:] if output else "(no output)")

    report = "\n".join(report_lines)
    print(report)
    Path(args.save_report).write_text(report, encoding="utf-8")
    print(f"\n[INFO] Report saved to {args.save_report}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
