"""Noiseless verification of each Pareto cut point at 60q.

Runs the cut + HEinsum + CudaQBackend(simulate=True) pipeline on every
selected Pareto point whose sub-circuits fit in state-vector sim
(max_size ≤ 20). Skips larger points — those go to HW untested.

Usage
-----
    uv run python -m evaluation.hardware.verify_pareto
"""
from __future__ import annotations

import sys
from time import perf_counter

import torch

from qtpu.compiler.opt import get_pareto_frontier
from qtpu.runtime import HEinsumRuntime
from qtpu.runtime.backends import CudaQBackend
from qtpu.transforms import circuit_to_heinsum
from evaluation.hardware.clifford_qnn import build_clifford_qnn_conjugated
from evaluation.hardware.run_pareto import (
    CIRCUIT_SIZE,
    SELECTED_MAX_SIZES,
    N_TRIALS,
    SEED,
    COMPILER_SEED,
    select_frontier_subset,
)


TOL = 1e-6
SIM_LIMIT = 20  # qubits: state-vector sim cutoff


def main():
    qc = build_clifford_qnn_conjugated(CIRCUIT_SIZE, seed=SEED)
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
        f"Selected Pareto points: {[p.max_size for p in frontier]}",
        flush=True,
    )

    fails = []
    for p in frontier:
        if p.max_size > SIM_LIMIT:
            print(
                f"  max_size={p.max_size:<3} SKIP (> {SIM_LIMIT}q, not state-vector simulable)",
                flush=True,
            )
            continue
        cut_circuit = opt_result.get_cut_circuit(p)
        htn = circuit_to_heinsum(cut_circuit)
        backend = CudaQBackend(simulate=True, estimate_qpu_time=False)
        rt = HEinsumRuntime(
            htn, backend=backend, dtype=torch.float64, device=torch.device("cpu")
        )
        rt.prepare(optimize=False)
        t0 = perf_counter()
        res, _ = rt.execute()
        t_exec = perf_counter() - t0
        val = float(res.item() if res.ndim == 0 else res.sum().item())
        status = "OK" if abs(val - 1.0) <= TOL else "FAIL"
        print(
            f"  max_size={p.max_size:<3} c_cost={p.c_cost:.2e}  parts={len(htn.quantum_tensors):<3} "
            f"exec={t_exec:.1f}s  recon={val:+.6f} [{status}]",
            flush=True,
        )
        if status == "FAIL":
            fails.append(p.max_size)
    if fails:
        print(f"\nFAIL: {fails}")
        sys.exit(1)
    print("\nOK: all simulable Pareto points reconstruct to +1")


if __name__ == "__main__":
    main()
