#!/usr/bin/env python
"""Reproduce all qTPU OSDI '26 evaluation results.

Usage:
    python -m evaluation.run_all                 # full evaluation (~2 h)
    python -m evaluation.run_all --quick         # smoke test (~5 min)
    python -m evaluation.run_all --plots-only    # regenerate plots from logs
    python -m evaluation.run_all --fig 14        # single figure

No QPU or GPU required. All quantum execution times are estimated
via Qiskit transpilation to IBM FakeMarrakesh (ASAP scheduling).

Output:
    logs/          JSONL benchmark data
    plots/         PDF figures matching the paper
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
ROOT = Path(__file__).resolve().parent.parent

# ─── Figure-to-script mapping ───────────────────────────────────────────────

FIGURES: dict[str, dict] = {
    # ── Compiler ────────────────────────────────────────────────────────────
    "9-10": {
        "title": "Figs 9, 10, Table 1 — Compiler tradeoff & scalability",
        "run": [
            ("Compiler QTPU", [PY, "-m", "evaluation.compiler.run", "qtpu"]),
            ("Compiler QAC",  [PY, "-m", "evaluation.compiler.run", "qac"]),
        ],
        "plot": ("Compiler plots", [PY, "-m", "evaluation.compiler.plot"]),
        "outputs": [
            "plots/pareto_frontiers.pdf",   # Fig 9
            "plots/scalability.pdf",        # Fig 10
            "plots/compile_times.pdf",      # Table 1
        ],
        "time": "~30 min",
    },

    # ── Runtime ─────────────────────────────────────────────────────────────
    "11": {
        "title": "Fig 11 — Runtime analysis",
        "run": [
            ("Runtime standard", [PY, "-m", "evaluation.runtime.run", "standard"]),
            ("Runtime dist-VQE", [PY, "-m", "evaluation.runtime.run", "dist"]),
        ],
        "plot": ("Runtime plots", [PY, "-m", "evaluation.runtime.plot"]),
        "outputs": ["plots/runtime_analysis.pdf"],
        "time": "~40 min",
    },

    # ── Scale / Circuit Knitting ────────────────────────────────────────────
    "12": {
        "title": "Fig 12 — Scalable hybrid computing (circuit knitting)",
        "run": [
            ("Scale QTPU", [PY, "-m", "evaluation.use_cases.scale.run", "qtpu"]),
            ("Scale QAC",  [PY, "-m", "evaluation.use_cases.scale.run", "qac"]),
        ],
        "plot": ("Scale plots", [PY, "-m", "evaluation.use_cases.scale.plot"]),
        "outputs": ["plots/scale_qnn.pdf"],
        "time": "~30 min",
    },

    # ── Hybrid ML ───────────────────────────────────────────────────────────
    "13": {
        "title": "Fig 13 — Hybrid machine learning",
        "run": [
            ("Hybrid ML", [PY, "-m", "evaluation.use_cases.hybrid_ml.run", "all"]),
        ],
        "plot": ("Hybrid ML plots", [PY, "-m", "evaluation.use_cases.hybrid_ml.plot"]),
        "outputs": ["plots/hybrid_ml/benchmark.pdf"],
        "time": "~20 min",
    },

    # ── Error Mitigation ────────────────────────────────────────────────────
    "14": {
        "title": "Fig 14 — Quantum error mitigation",
        "run": [
            ("Error mitigation", [PY, "-m", "evaluation.use_cases.error_mitigation.run", "all"]),
        ],
        "plot": ("Error mitigation plots", [PY, "-m", "evaluation.use_cases.error_mitigation.plot"]),
        "outputs": ["plots/error_mitigation.pdf"],
        "time": "~3 min",
    },
}

# Order for full evaluation (fastest first for quick feedback)
FIGURE_ORDER = ["14", "13", "12", "9-10", "11"]


def run_step(name: str, cmd: list[str]) -> bool:
    """Run a single step. Returns True on success."""
    print(f"\n{'─' * 60}")
    print(f"  {name}")
    print(f"{'─' * 60}\n", flush=True)
    t0 = time.time()
    ret = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0
    if ret.returncode != 0:
        print(f"\n  FAILED: {name} (after {elapsed:.0f}s)\n")
        return False
    print(f"  Done: {name} ({elapsed:.0f}s)")
    return True


def run_figure(fig_key: str, *, plots_only: bool = False) -> bool:
    """Run benchmarks and plots for one figure group."""
    fig = FIGURES[fig_key]
    print(f"\n{'=' * 60}")
    print(f"  {fig['title']}  (est. {fig['time']})")
    print(f"{'=' * 60}")

    ok = True
    if not plots_only:
        for name, cmd in fig["run"]:
            if not run_step(name, cmd):
                ok = False

    name, cmd = fig["plot"]
    if not run_step(name, cmd):
        ok = False

    # Check outputs
    for path in fig["outputs"]:
        if Path(path).exists():
            print(f"  -> {path}")
        else:
            print(f"  -> MISSING: {path}")

    return ok


def main() -> None:
    args = sys.argv[1:]
    plots_only = "--plots-only" in args
    quick = "--quick" in args

    # Single figure mode: --fig 14
    if "--fig" in args:
        idx = args.index("--fig")
        fig_key = args[idx + 1] if idx + 1 < len(args) else None
        if fig_key not in FIGURES:
            print(f"Unknown figure: {fig_key}")
            print(f"Available: {', '.join(FIGURES)}")
            sys.exit(1)
        ok = run_figure(fig_key, plots_only=plots_only)
        sys.exit(0 if ok else 1)

    # Quick smoke test: only Fig 14 (fastest, ~3 min)
    if quick:
        print("Quick smoke test: running Fig 14 only (~3 min)")
        ok = run_figure("14", plots_only=plots_only)
        sys.exit(0 if ok else 1)

    # Full evaluation
    t0 = time.time()
    failed = []
    for fig_key in FIGURE_ORDER:
        if not run_figure(fig_key, plots_only=plots_only):
            failed.append(fig_key)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  Completed in {elapsed/60:.0f} minutes")
    if failed:
        print(f"  Failed figures: {failed}")
    else:
        print("  All figures reproduced successfully.")
    print(f"{'=' * 60}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
