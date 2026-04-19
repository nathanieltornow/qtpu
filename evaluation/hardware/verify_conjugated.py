"""Verify the Clifford-conjugated QNN reconstructs to +1 through the
qtpu cut + HEinsum + CudaQBackend(simulate=True) pipeline.

Covers the sweep sizes we plan to run on hardware, using max_size=15 so
no sub-circuit exceeds 15 qubits.

Usage
-----
    uv run python -m evaluation.hardware.verify_conjugated
"""
from __future__ import annotations

import sys
from time import perf_counter

import torch

import qtpu
from qtpu.runtime import HEinsumRuntime
from qtpu.runtime.backends import CudaQBackend
from evaluation.hardware.clifford_qnn import build_clifford_qnn_conjugated


TOL = 1e-6
QPU_SIZE = 15
SIZES = [8, 12, 16, 20, 30, 40, 60]


def verify(n: int) -> dict:
    qc = build_clifford_qnn_conjugated(n, seed=42)
    t0 = perf_counter()
    cut = qtpu.cut(
        qc, max_size=QPU_SIZE, cost_weight=1000, n_trials=20, seed=1, num_workers=1
    )
    htn = qtpu.circuit_to_heinsum(cut)
    t_cut = perf_counter() - t0

    max_sub_q = max(qt._circuit.num_qubits for qt in htn.quantum_tensors)
    parts = len(htn.quantum_tensors)

    backend = CudaQBackend(simulate=True, estimate_qpu_time=False)
    rt = HEinsumRuntime(
        htn, backend=backend, dtype=torch.float64, device=torch.device("cpu")
    )
    rt.prepare(optimize=False)
    t0 = perf_counter()
    res, _ = rt.execute()
    t_exec = perf_counter() - t0
    val = float(res.item() if res.ndim == 0 else res.sum().item())

    return {
        "n": n,
        "parts": parts,
        "max_sub_q": max_sub_q,
        "recon": val,
        "t_cut": t_cut,
        "t_exec": t_exec,
    }


def main():
    sizes = (
        [int(s) for s in sys.argv[1].split(",")]
        if len(sys.argv) > 1 else SIZES
    )
    print(f"Verifying conjugated Clifford-QNN, QPU={QPU_SIZE}, sizes={sizes}", flush=True)
    fails = []
    for n in sizes:
        try:
            r = verify(n)
        except Exception as e:
            print(f"  n={n:<3} ERROR: {e}", flush=True)
            fails.append(n)
            continue
        status = "OK" if abs(r["recon"] - 1.0) <= TOL else "FAIL"
        print(
            f"  n={r['n']:<3} parts={r['parts']:<3} max_sub_q={r['max_sub_q']:<3} "
            f"cut={r['t_cut']:.1f}s  exec={r['t_exec']:.1f}s  "
            f"recon={r['recon']:+.6f} [{status}]",
            flush=True,
        )
        if status == "FAIL":
            fails.append(n)
    if fails:
        print(f"\nFAIL: {fails}")
        sys.exit(1)
    print("\nOK: all sizes reconstruct to +1 through cut + HEinsum pipeline")


if __name__ == "__main__":
    main()
