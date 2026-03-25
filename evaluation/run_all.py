#!/usr/bin/env python
"""Run all qTPU OSDI evaluations and generate plots.

Usage:
    python -m evaluation.run_all              # run everything
    python -m evaluation.run_all --plots-only # re-generate plots from existing logs
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BENCHMARKS = [
    # ── Compiler (Figs 9, 10, Table 1) ──────────────────────────────
    ("Compiler · QTPU",          [sys.executable, "-m", "evaluation.compiler.run", "qtpu"]),
    ("Compiler · QAC",           [sys.executable, "-m", "evaluation.compiler.run", "qac"]),

    # ── Runtime (Fig 11) ────────────────────────────────────────────
    ("Runtime · Standard",       [sys.executable, "-m", "evaluation.runtime.run", "standard"]),
    ("Runtime · Dist-VQE",       [sys.executable, "-m", "evaluation.runtime.run", "dist"]),

    # ── Circuit Knitting / Scale (Fig 12) ───────────────────────────
    ("Scale · QTPU",             [sys.executable, "-m", "evaluation.use_cases.scale.run", "qtpu"]),
    ("Scale · QAC",              [sys.executable, "-m", "evaluation.use_cases.scale.run", "qac"]),

    # ── Hybrid ML (Fig 13) ──────────────────────────────────────────
    ("Hybrid ML",                [sys.executable, "-m", "evaluation.use_cases.hybrid_ml.run", "all"]),

    # ── Error Mitigation (Fig 14) ───────────────────────────────────
    ("Error Mitigation",         [sys.executable, "-m", "evaluation.use_cases.error_mitigation.run", "all"]),
]

PLOTS = [
    ("Plot · Compiler",          [sys.executable, "-m", "evaluation.compiler.plot"]),
    ("Plot · Runtime",           [sys.executable, "-m", "evaluation.runtime.plot"]),
    ("Plot · Scale",             [sys.executable, "-m", "evaluation.use_cases.scale.plot"]),
    ("Plot · Hybrid ML",         [sys.executable, "-m", "evaluation.use_cases.hybrid_ml.plot"]),
    ("Plot · Error Mitigation",  [sys.executable, "-m", "evaluation.use_cases.error_mitigation.plot"]),
]


def run_steps(steps: list[tuple[str, list[str]]]) -> list[str]:
    """Run a list of (name, command) steps, returning names of failures."""
    failed = []
    for name, cmd in steps:
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}\n")
        ret = subprocess.run(cmd, cwd=ROOT)
        if ret.returncode != 0:
            failed.append(name)
            print(f"\n  *** FAILED: {name} ***\n")
    return failed


def main() -> None:
    plots_only = "--plots-only" in sys.argv

    failed: list[str] = []

    if not plots_only:
        failed += run_steps(BENCHMARKS)

    failed += run_steps(PLOTS)

    print("\n" + "=" * 60)
    if failed:
        print(f"  Done with {len(failed)} failure(s): {failed}")
        sys.exit(1)
    else:
        print("  All evaluations completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
