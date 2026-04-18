"""Hardware Pareto: classical-vs-quantum cost tradeoff on IBM Marrakesh.

Sweeps the QPU-size knob (cutting aggressiveness) at a fixed 60-qubit
Clifford-QNN mirror circuit. The circuit's ideal ⟨Z^⊗n⟩ is +1 (the
mirror collapses to the identity under unitary reconstruction), which
gives a strong signal that hardware noise degrades toward 0 — unlike a
random QNN where noiseless and noisy both average to ≈0 and fidelity
becomes unmeasurable.

Pareto shape:
    QPU=5   → many small partitions → highest classical cost, highest
              hardware fidelity
    QPU=10  → fewer, larger partitions
    QPU=20
    QPU=30
    QPU=60  → monolithic (uncut) → zero classical cost, lowest
              hardware fidelity

Usage
-----
    uv run python -m evaluation.hardware.run_pareto
    uv run python -m evaluation.hardware.run_pareto 10,60
"""

from __future__ import annotations

import os
import sys
from time import perf_counter

import torch

import benchkit as bk

import qtpu
from qtpu.runtime import HEinsumRuntime
from qtpu.runtime.ibm_backend import IBMBackend
from evaluation.analysis import estimate_runtime
from evaluation.hardware.clifford_qnn import build_clifford_qnn_mirror


try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass


CIRCUIT_SIZE = 60
QPU_SIZES = [5, 10, 20, 30, 60]  # 60 = monolithic / uncut endpoint
SHOTS = 1000
BACKEND_NAME = "ibm_marrakesh"
LOG_PATH = "logs/hardware/pareto.jsonl"


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
    qc_m.measure_all()
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


def _submit_cut(qc, device, qpu_size: int):
    """Cut with qtpu, then run flat subcircuits via HEinsumRuntime/IBMBackend."""
    compile_start = perf_counter()
    cut_circuit = qtpu.cut(
        qc,
        max_size=qpu_size,
        cost_weight=1000,
        n_trials=20,
        seed=42,
        num_workers=1,
    )
    htn = qtpu.circuit_to_heinsum(cut_circuit)
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


def run_point(qpu_size: int, device):
    qc = build_clifford_qnn_mirror(CIRCUIT_SIZE, seed=42)
    ideal = 1.0
    print(
        f"\n[Pareto] circuit_size={CIRCUIT_SIZE}q  QPU={qpu_size}q  "
        f"({'monolithic' if qpu_size >= CIRCUIT_SIZE else 'cut'})",
        flush=True,
    )

    if qpu_size >= CIRCUIT_SIZE:
        inner = _submit_monolithic(qc, device)
    else:
        inner = _submit_cut(qc, device, qpu_size)

    err = abs(inner["hw_expval"] - ideal)
    fidelity = max(0.0, 1.0 - err / 2.0)
    out = {
        "backend_name": f"ibm-{device.name}",
        "circuit_size": CIRCUIT_SIZE,
        "qpu_size": qpu_size,
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


def main():
    if len(sys.argv) > 1:
        qpus = [int(s) for s in sys.argv[1].split(",")]
    else:
        qpus = QPU_SIZES
    print(f"Pareto sweep @ {CIRCUIT_SIZE}q Clifford-QNN mirror, QPU sizes: {qpus}", flush=True)

    device = _connect(BACKEND_NAME)
    status = device.status()
    print(
        f"Backend: {device.name} operational={status.operational} "
        f"pending={status.pending_jobs}",
        flush=True,
    )

    @bk.foreach(qpu_size=qpus)
    @bk.log(LOG_PATH)
    def bench(qpu_size: int):
        return run_point(qpu_size=qpu_size, device=device)

    bench()


if __name__ == "__main__":
    main()
