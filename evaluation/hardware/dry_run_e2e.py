"""Budget-only dry-run of the end-to-end sweep — no execution, no simulation.

Mirrors `run.py` up to circuit compilation, then reports the HW-budget
numbers (flat-circuit count, FakeMarrakesh ASAP QPU-time estimate) per
size. Use this to sanity-check the HW budget before submitting to IBM.

Usage
-----
    uv run python -m evaluation.hardware.dry_run_e2e
    uv run python -m evaluation.hardware.dry_run_e2e 20,40
"""
from __future__ import annotations

import sys
from time import perf_counter

import qtpu
from evaluation.analysis import estimate_runtime
from evaluation.hardware.clifford_qnn import build_clifford_qnn_conjugated
from evaluation.hardware.run import DEFAULT_SIZES, QPU_SIZE


def dry_run_size(n: int) -> dict:
    qc = build_clifford_qnn_conjugated(n, seed=42)

    t0 = perf_counter()
    cut_circuit = qtpu.cut(
        qc, max_size=QPU_SIZE, cost_weight=1000, n_trials=20, seed=1, num_workers=1
    )
    htn = qtpu.circuit_to_heinsum(cut_circuit)
    t_cut = perf_counter() - t0

    flat_circuits = [c.decompose() for qt in htn.quantum_tensors for c in qt.flat()]
    estimated_qpu = estimate_runtime(flat_circuits)

    return {
        "size": n,
        "num_partitions": len(htn.quantum_tensors),
        "num_flat_circuits": len(flat_circuits),
        "estimated_qpu_time": estimated_qpu,
        "t_cut": t_cut,
    }


def main():
    if len(sys.argv) > 1:
        sizes = [int(s) for s in sys.argv[1].split(",")]
    else:
        sizes = DEFAULT_SIZES

    print(
        f"Dry-run end-to-end sweep @ Clifford-QNN mirror, QPU={QPU_SIZE}, sizes={sizes}",
        flush=True,
    )

    total_circuits = 0
    total_est_qpu = 0.0
    for n in sizes:
        r = dry_run_size(n)
        total_circuits += r["num_flat_circuits"]
        total_est_qpu += r["estimated_qpu_time"]
        print(
            f"  n={r['size']:<3} parts={r['num_partitions']:<3} "
            f"flat_circs={r['num_flat_circuits']:<5} "
            f"est_qpu={r['estimated_qpu_time']:.2f}s  "
            f"(cut {r['t_cut']:.1f}s)",
            flush=True,
        )

    print()
    print("=" * 72)
    print(f"HW BUDGET SUMMARY ({len(sizes)} sizes)")
    print("=" * 72)
    print(f"  total flat sub-circuits across sizes: {total_circuits}")
    print(f"  total estimated QPU time (sum):       {total_est_qpu:.1f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()
