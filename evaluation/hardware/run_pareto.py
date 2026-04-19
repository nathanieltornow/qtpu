"""Hardware Pareto: classical-vs-quantum cost tradeoff on IBM Marrakesh.

At a fixed 60-qubit Clifford-QNN mirror circuit, enumerates the
automatically-generated Pareto frontier from the cut-compiler (each
point is a (c_cost, max_error, max_size) solution) and runs each point
on real hardware. Plus the monolithic (uncut) endpoint: zero classical
cost, lowest hardware fidelity.

The mirror circuit's ideal ⟨Z^⊗n⟩ is +1 (the mirror collapses to the
identity under unitary reconstruction), so hardware noise drives the
reconstructed expectation value toward 0 — a clean measurable fidelity
signal.

Usage
-----
    uv run python -m evaluation.hardware.run_pareto
"""

from __future__ import annotations

import os
import sys
from time import perf_counter

import torch

import benchkit as bk

import qtpu
from qtpu.compiler.opt import get_pareto_frontier, CutPoint
from qtpu.runtime import HEinsumRuntime
from qtpu.runtime.ibm_backend import IBMBackend
from qtpu.transforms import circuit_to_heinsum
from evaluation.analysis import estimate_runtime
from evaluation.hardware.clifford_qnn import build_clifford_qnn_conjugated


try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass


CIRCUIT_SIZE = 60
SHOTS = 1000
BACKEND_NAME = "ibm_marrakesh"
LOG_PATH = "logs/hardware/pareto.jsonl"
N_TRIALS = 20
SEED = 42              # circuit builder seed (fixes the Clifford-QNN instance)
COMPILER_SEED = 1      # cutter/Pareto seed (chosen from search_seeds.py: best geomean c_cost across all 5 targets)

# Sub-select frontier points at these max_size values (one point per size, cheapest
# c_cost variant) so the experiment stays within HW budget. These five were chosen
# to span ~7 orders of c_cost on the 60q Clifford mirror frontier. Plus monolithic
# (60q uncut) at the classical=0 endpoint.
SELECTED_MAX_SIZES = [10, 15, 31]


def select_frontier_subset(frontier, max_sizes: list[int]) -> list[CutPoint]:
    """Pick one frontier point per requested max_size. If the exact size
    is not present (frontier enumeration has some run-to-run variability),
    snap to the nearest available size; keep the cheapest c_cost at each
    size. De-duplicate if two requested sizes snap to the same point."""
    if not frontier:
        return []
    cheapest_by_size: dict[int, CutPoint] = {}
    for p in frontier:
        cur = cheapest_by_size.get(p.max_size)
        if cur is None or p.c_cost < cur.c_cost:
            cheapest_by_size[p.max_size] = p
    available = sorted(cheapest_by_size.keys())

    picked: dict[int, CutPoint] = {}
    for target in max_sizes:
        nearest = min(available, key=lambda s: (abs(s - target), s))
        picked[nearest] = cheapest_by_size[nearest]
    return [picked[s] for s in sorted(picked.keys())]


def _connect(backend_name: str):
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(
        channel="ibm_cloud",
        token=os.environ["IBM_TOKEN"],
        instance=os.environ["IBM_CRN"],
    )
    return service.backend(backend_name)


def _submit_monolithic(qc, device):
    """Submit the full uncut circuit via SamplerV2 (zero classical cost)."""
    from qiskit.compiler import transpile
    from qiskit_ibm_runtime import SamplerV2

    qc_m = qc.copy()
    # Measure only non-reset-terminal qubits (I-observable positions are
    # terminated with reset by build_clifford_qnn_conjugated). measure_all
    # would include them and let noisy flips corrupt the Z^n parity.
    from qtpu.runtime.ibm_backend import _measure_non_reset_qubits
    _measure_non_reset_qubits(qc_m)
    compile_start = perf_counter()
    transpiled = transpile(qc_m, backend=device, optimization_level=3)
    compile_time = perf_counter() - compile_start

    depth = transpiled.depth()
    n_2q = sum(1 for i in transpiled.data if i.operation.num_qubits == 2)
    print(f"    transpiled depth={depth}  2q_gates={n_2q}", flush=True)

    sampler = SamplerV2(mode=device)
    hw_start = perf_counter()
    job = sampler.run([transpiled], shots=SHOTS)
    result = job.result()
    hw_wall = perf_counter() - hw_start

    actual_qpu = job.metrics().get("usage", {}).get("quantum_seconds", 0.0)

    counts = result[0].data.meas.get_counts()
    total = sum(counts.values())
    hw_expval = sum(
        ((-1) ** bs.count("1")) * c / total
        for bs, c in counts.items()
    )

    return {
        "num_partitions": 1,
        "num_flat_circuits": 1,
        "num_jobs": 1,
        "compile_time": compile_time,
        "estimated_qpu_time": None,  # ASAP estimator is designed for flat subcircuits
        "actual_qpu_time": actual_qpu,
        "hw_expval": hw_expval,
        "hw_wall_time": hw_wall,
        "transpiled_depth": depth,
        "transpiled_2q_gates": n_2q,
    }


def _submit_cut(qc, device, point: CutPoint, opt_result):
    """Run a specific Pareto CutPoint on hardware via HEinsumRuntime/IBMBackend."""
    compile_start = perf_counter()
    cut_circuit = opt_result.get_cut_circuit(point)
    htn = circuit_to_heinsum(cut_circuit)
    compile_time = perf_counter() - compile_start

    flat_circuits = [c.decompose() for qt in htn.quantum_tensors for c in qt.flat()]
    estimated_qpu = estimate_runtime(flat_circuits)

    backend = IBMBackend(backend=device, shots=SHOTS)
    runtime = HEinsumRuntime(
        htn, backend=backend, dtype=torch.float64, device=torch.device("cpu")
    )
    runtime.prepare(optimize=False)

    hw_start = perf_counter()
    hw_result, _ = runtime.execute()
    hw_wall = perf_counter() - hw_start
    hw_expval = float(hw_result.item() if hw_result.ndim == 0 else hw_result.sum().item())

    return {
        "num_partitions": len(htn.quantum_tensors),
        "num_flat_circuits": len(flat_circuits),
        "num_jobs": backend.total_jobs,
        "compile_time": compile_time,
        "estimated_qpu_time": estimated_qpu,
        "actual_qpu_time": backend.total_actual_qpu_time,
        "hw_expval": hw_expval,
        "hw_wall_time": hw_wall,
    }


def run_frontier_point(point: CutPoint, qc, opt_result, device):
    """Run one Pareto frontier point on hardware."""
    ideal = 1.0
    print(
        f"\n[Pareto] max_size={point.max_size}  c_cost={point.c_cost:.2e}  "
        f"max_error={point.max_error:.4f}  sampling_cost={point.sampling_cost:.2f}",
        flush=True,
    )

    inner = _submit_cut(qc, device, point, opt_result)

    err = abs(inner["hw_expval"] - ideal)
    fidelity = max(0.0, 1.0 - err / 2.0)
    out = {
        "backend_name": f"ibm-{device.name}",
        "circuit_size": CIRCUIT_SIZE,
        "qpu_size": point.max_size,
        "c_cost": point.c_cost,
        "max_error_estimate": point.max_error,
        "sampling_cost": point.sampling_cost,
        "mode": "cut",
        "ideal_expval": ideal,
        "abs_error": err,
        "fidelity": fidelity,
        **inner,
    }
    print(
        f"  parts={out['num_partitions']:<3} circs={out['num_flat_circuits']:<4} "
        f"jobs={out['num_jobs']:<2} "
        f"actual_qpu={out['actual_qpu_time']}s  "
        f"hw_expval={out['hw_expval']:+.4f}  "
        f"fidelity={fidelity:.4f}",
        flush=True,
    )
    return out


def run_monolithic(qc, device):
    """Run the monolithic (uncut) endpoint."""
    ideal = 1.0
    print(
        f"\n[Pareto] monolithic (uncut) {CIRCUIT_SIZE}q",
        flush=True,
    )
    inner = _submit_monolithic(qc, device)
    err = abs(inner["hw_expval"] - ideal)
    fidelity = max(0.0, 1.0 - err / 2.0)
    out = {
        "backend_name": f"ibm-{device.name}",
        "circuit_size": CIRCUIT_SIZE,
        "qpu_size": CIRCUIT_SIZE,
        "c_cost": 0.0,
        "max_error_estimate": None,
        "sampling_cost": 0.0,
        "mode": "monolithic",
        "ideal_expval": ideal,
        "abs_error": err,
        "fidelity": fidelity,
        **inner,
    }
    print(
        f"  hw_expval={out['hw_expval']:+.4f}  fidelity={fidelity:.4f}",
        flush=True,
    )
    return out


def main():
    print(
        f"Pareto sweep @ {CIRCUIT_SIZE}q Clifford-QNN mirror, "
        f"enumerating compiler Pareto frontier",
        flush=True,
    )

    qc = build_clifford_qnn_conjugated(CIRCUIT_SIZE, seed=SEED)

    # Enumerate the full Pareto frontier once
    t0 = perf_counter()
    opt_result = get_pareto_frontier(
        qc,
        max_sampling_cost=150,
        num_workers=1,
        n_trials=N_TRIALS,
        seed=COMPILER_SEED,
    )
    # The frontier may include a max_size=CIRCUIT_SIZE point (no cuts, c_cost=0);
    # we handle that as "monolithic" below via SamplerV2 so we don't double-run it.
    full_frontier = sorted(
        (p for p in opt_result.pareto_frontier if p.max_size < CIRCUIT_SIZE),
        key=lambda p: p.max_size,
    )
    frontier = select_frontier_subset(full_frontier, SELECTED_MAX_SIZES)
    t_frontier = perf_counter() - t0
    print(
        f"Pareto frontier: {len(full_frontier)} cut points enumerated in "
        f"{t_frontier:.1f}s; running subset of {len(frontier)} "
        f"(max_sizes={SELECTED_MAX_SIZES}) plus monolithic",
        flush=True,
    )
    for p in frontier:
        print(
            f"  max_size={p.max_size:<3} c_cost={p.c_cost:.2e}  "
            f"max_error={p.max_error:.4f}  sampling_cost={p.sampling_cost:.2f}",
            flush=True,
        )

    device = _connect(BACKEND_NAME)
    status = device.status()
    print(
        f"Backend: {device.name} operational={status.operational} "
        f"pending={status.pending_jobs}",
        flush=True,
    )

    point_indices = list(range(len(frontier))) + ["monolithic"]

    @bk.foreach(point_idx=point_indices)
    @bk.log(LOG_PATH)
    def bench(point_idx):
        if point_idx == "monolithic":
            return run_monolithic(qc, device)
        return run_frontier_point(frontier[point_idx], qc, opt_result, device)

    bench()


if __name__ == "__main__":
    main()
