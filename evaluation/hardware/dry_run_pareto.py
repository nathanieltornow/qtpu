"""Budget-only dry-run of the Pareto sweep — no execution, no simulation.

Mirrors `run_pareto.py` up to circuit compilation, then reports the
HW-budget numbers (flat-circuit count, FakeMarrakesh ASAP QPU-time
estimate) per selected Pareto point. Use this to sanity-check the HW
budget before submitting to IBM.

Usage
-----
    uv run python -m evaluation.hardware.dry_run_pareto
"""
from __future__ import annotations

from time import perf_counter

from qtpu.compiler.opt import get_pareto_frontier, CutPoint
from qtpu.transforms import circuit_to_heinsum
from evaluation.analysis import estimate_runtime
from evaluation.hardware.clifford_qnn import build_clifford_qnn_conjugated
from evaluation.hardware.run_pareto import (
    CIRCUIT_SIZE,
    SELECTED_MAX_SIZES,
    N_TRIALS,
    SEED,
    COMPILER_SEED,
    select_frontier_subset,
)


def dry_run_point(point: CutPoint, opt_result) -> dict:
    t0 = perf_counter()
    cut_circuit = opt_result.get_cut_circuit(point)
    htn = circuit_to_heinsum(cut_circuit)
    t_cut = perf_counter() - t0

    flat_circuits = [c.decompose() for qt in htn.quantum_tensors for c in qt.flat()]
    estimated_qpu = estimate_runtime(flat_circuits)

    return {
        "max_size": point.max_size,
        "c_cost": point.c_cost,
        "max_error_est": point.max_error,
        "sampling_cost": point.sampling_cost,
        "num_partitions": len(htn.quantum_tensors),
        "num_flat_circuits": len(flat_circuits),
        "estimated_qpu_time": estimated_qpu,
        "t_cut": t_cut,
    }


def main():
    print(
        f"Dry-run Pareto @ {CIRCUIT_SIZE}q Clifford-QNN mirror, "
        f"selected max_sizes={SELECTED_MAX_SIZES}",
        flush=True,
    )

    qc = build_clifford_qnn_conjugated(CIRCUIT_SIZE, seed=SEED)

    t0 = perf_counter()
    opt_result = get_pareto_frontier(
        qc,
        max_sampling_cost=150,
        num_workers=1,
        n_trials=N_TRIALS,
        seed=COMPILER_SEED,
    )
    full_frontier = sorted(
        (p for p in opt_result.pareto_frontier if p.max_size < CIRCUIT_SIZE),
        key=lambda p: p.max_size,
    )
    frontier = select_frontier_subset(full_frontier, SELECTED_MAX_SIZES)
    print(
        f"Pareto frontier: {len(full_frontier)} cut points in "
        f"{perf_counter() - t0:.1f}s; running subset of {len(frontier)}",
        flush=True,
    )

    rows = []
    total_circuits = 0
    total_est_qpu = 0.0
    for p in frontier:
        r = dry_run_point(p, opt_result)
        rows.append(r)
        total_circuits += r["num_flat_circuits"]
        total_est_qpu += r["estimated_qpu_time"]
        print(
            f"  max_size={r['max_size']:<3} c_cost={r['c_cost']:.2e}  "
            f"parts={r['num_partitions']:<3} flat_circs={r['num_flat_circuits']:<5} "
            f"est_qpu={r['estimated_qpu_time']:.2f}s  "
            f"(cut {r['t_cut']:.1f}s)",
            flush=True,
        )

    print()
    print("=" * 72)
    print(f"HW BUDGET SUMMARY ({len(frontier)} cut points + 1 monolithic)")
    print("=" * 72)
    print(f"  total flat sub-circuits across cut points: {total_circuits}")
    print(f"  total estimated QPU time (sum of points):  {total_est_qpu:.1f}s")
    print(f"  + 1 monolithic {CIRCUIT_SIZE}q circuit (1 job)")
    print("=" * 72)


if __name__ == "__main__":
    main()
