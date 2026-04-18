"""Real-Hardware Benchmark on IBM Marrakesh (OSDI Condition 2)
================================================================

Runs cut QNN subcircuits on IBM Marrakesh via qiskit-ibm-runtime, measures
actual QPU time from the job metadata, and compares against the
FakeMarrakesh ASAP-schedule estimate from ``estimate_runtime``. Also
compares the reconstructed expectation value against a noiseless Aer/CUDA-Q
simulation to report fidelity for each circuit size.

Requires IBM_TOKEN and IBM_CRN in the environment (or a .env file).

Usage
-----
    uv run python -m evaluation.hardware.run
    uv run python -m evaluation.hardware.run 20,40    # restrict sizes
"""

from __future__ import annotations

import os
import sys
from time import perf_counter

import torch
from qiskit.circuit import QuantumCircuit

import benchkit as bk

import qtpu
from qtpu.runtime import HEinsumRuntime
from qtpu.runtime.backends import CudaQBackend
from qtpu.runtime.ibm_backend import IBMBackend
from evaluation.analysis import estimate_runtime


try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass


# =============================================================================
# Configuration
# =============================================================================

# Primary sweep (kept small enough for real IBM queue budgets).
DEFAULT_SIZES = [20, 40, 60]
QPU_SIZE = 10
SHOTS = 1000
BACKEND_NAME = "ibm_marrakesh"
LOG_PATH = "logs/hardware/ibm.jsonl"


def _connect(backend_name: str):
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(
        channel="ibm_cloud",
        token=os.environ["IBM_TOKEN"],
        instance=os.environ["IBM_CRN"],
    )
    return service.backend(backend_name)


def noiseless_expval(htn) -> float:
    """Run the cut hTN under noiseless CUDA-Q to get the ideal scalar."""
    sim_backend = CudaQBackend(target="qpp-cpu", simulate=True, estimate_qpu_time=False)
    runtime = HEinsumRuntime(
        htn, backend=sim_backend, dtype=torch.float64, device=torch.device("cpu")
    )
    runtime.prepare(optimize=False)
    result, _ = runtime.execute()
    return float(result.item() if result.ndim == 0 else result.sum().item())


def run_hw(circuit_size: int, device):
    """Build, cut, estimate, run on real IBM hardware, and compare."""
    from mqt.bench import get_benchmark_indep

    print(f"\n[HW ibm] size={circuit_size}q → QPU={QPU_SIZE}q", flush=True)

    # Build + cut
    qc = get_benchmark_indep("qnn", circuit_size=circuit_size, opt_level=3)
    qc = qc.remove_final_measurements(inplace=False)
    compile_start = perf_counter()
    cut_circuit = qtpu.cut(
        qc, max_size=QPU_SIZE, cost_weight=1000, n_trials=20, seed=42, num_workers=1
    )
    htn = qtpu.circuit_to_heinsum(cut_circuit)
    compile_time = perf_counter() - compile_start

    # Flat circuits for FakeMarrakesh estimation
    flat_circuits = [c.decompose() for qt in htn.quantum_tensors for c in qt.flat()]
    est_start = perf_counter()
    estimated_qpu_time = estimate_runtime(flat_circuits)
    estimate_time = perf_counter() - est_start

    # Run on IBM hardware
    backend = IBMBackend(backend=device, shots=SHOTS)
    runtime = HEinsumRuntime(
        htn, backend=backend, dtype=torch.float64, device=torch.device("cpu")
    )
    runtime.prepare(optimize=False)

    hw_start = perf_counter()
    hw_result, _timing = runtime.execute()
    hw_wall_time = perf_counter() - hw_start
    hw_val = float(hw_result.item() if hw_result.ndim == 0 else hw_result.sum().item())

    # Noiseless reference
    ref_val = noiseless_expval(htn)

    # Fidelity for a scalar expectation value in [-1, 1]: 1 - |err|/2
    fidelity = 1.0 - abs(hw_val - ref_val) / 2.0

    result = {
        "backend_name": backend.name,
        "num_partitions": len(htn.quantum_tensors),
        "num_flat_circuits": len(flat_circuits),
        "compile_time": compile_time,
        "estimate_time": estimate_time,
        "estimated_qpu_time": estimated_qpu_time,
        "actual_qpu_time": backend.total_actual_qpu_time,
        "num_jobs": backend.total_jobs,
        "hw_wall_time": hw_wall_time,
        "hw_expval": hw_val,
        "noiseless_expval": ref_val,
        "abs_error": abs(hw_val - ref_val),
        "fidelity": fidelity,
        "qpu_time_ratio": (
            backend.total_actual_qpu_time / estimated_qpu_time
            if estimated_qpu_time > 0 else None
        ),
    }
    print(
        f"  partitions={result['num_partitions']} "
        f"flat={result['num_flat_circuits']} "
        f"est_qpu={estimated_qpu_time:.3f}s "
        f"actual_qpu={result['actual_qpu_time']:.3f}s "
        f"ratio={result['qpu_time_ratio']} "
        f"err={result['abs_error']:.4f} fidelity={fidelity:.4f}",
        flush=True,
    )
    return result


def main():
    if len(sys.argv) > 1:
        sizes = [int(s) for s in sys.argv[1].split(",")]
    else:
        sizes = DEFAULT_SIZES
    print(f"Sizes: {sizes}", flush=True)

    # Connect once, reuse across sizes
    device = _connect(BACKEND_NAME)
    status = device.status()
    print(f"Backend: {device.name} operational={status.operational} "
          f"pending={status.pending_jobs}", flush=True)

    @bk.foreach(circuit_size=sizes)
    @bk.log(LOG_PATH)
    def bench(circuit_size: int):
        return run_hw(circuit_size=circuit_size, device=device)

    bench()


if __name__ == "__main__":
    main()
