"""Run qTPU end-to-end on IBM Marrakesh.

Cuts a QNN circuit, runs the subcircuits on real hardware via IBMBackend,
and performs the full hTN contraction to get the reconstructed result.
Compares against the FakeMarrakesh QPU time estimate.
"""

import os
from time import perf_counter

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

# --- Config ---
CIRCUIT_SIZE = 20
QPU_SIZE = 10
BACKEND_NAME = "ibm_marrakesh"
SHOTS = 1000


def main():
    from mqt.bench import get_benchmark_indep
    from qiskit_ibm_runtime import QiskitRuntimeService

    import qtpu
    from qtpu.runtime import HEinsumRuntime
    from qtpu.runtime.ibm_backend import IBMBackend
    from evaluation.analysis import estimate_runtime

    # --- Build and cut circuit ---
    print(f"Building {CIRCUIT_SIZE}-qubit QNN, cutting to {QPU_SIZE}-qubit subcircuits...")
    qc = get_benchmark_indep("qnn", circuit_size=CIRCUIT_SIZE, opt_level=3)
    qc = qc.remove_final_measurements(inplace=False)

    compile_start = perf_counter()
    cut_circuit = qtpu.cut(
        qc, max_size=QPU_SIZE, cost_weight=1000, n_trials=20, seed=42, num_workers=1
    )
    htn = qtpu.circuit_to_heinsum(cut_circuit)
    compile_time = perf_counter() - compile_start

    print(f"Compile time: {compile_time:.1f}s")
    print(f"Subcircuits: {len(htn.quantum_tensors)}")
    for i, qt in enumerate(htn.quantum_tensors):
        print(f"  QT[{i}]: shape={qt.shape}, qubits={qt.circuit.num_qubits}")

    # --- Estimate QPU time (FakeMarrakesh) ---
    print("\nEstimating QPU time (FakeMarrakesh)...")
    flat_circuits = []
    for qt in htn.quantum_tensors:
        flat_circuits.extend([c.decompose() for c in qt.flat()])
    estimated_qpu_time = estimate_runtime(flat_circuits)
    print(f"Estimated QPU time: {estimated_qpu_time:.4f}s")
    print(f"Total flat circuits: {len(flat_circuits)}")

    # --- Connect to IBM and run ---
    print(f"\nConnecting to {BACKEND_NAME}...")
    service = QiskitRuntimeService(
        channel="ibm_cloud",
        token=os.environ["IBM_TOKEN"],
        instance=os.environ["IBM_CRN"],
    )
    ibm_backend_device = service.backend(BACKEND_NAME)
    print(f"Connected: {ibm_backend_device.name} ({ibm_backend_device.num_qubits} qubits)")

    # Create IBMBackend and HEinsumRuntime
    backend = IBMBackend(backend=ibm_backend_device, shots=SHOTS)
    runtime = HEinsumRuntime(
        htn, backend=backend, dtype=torch.float64, device=torch.device("cpu")
    )

    print("\nPreparing runtime...")
    runtime.prepare(optimize=False)

    print(f"Running full hTN pipeline on {BACKEND_NAME}...")
    run_start = perf_counter()
    result, timing = runtime.execute()
    run_time = perf_counter() - run_start

    # --- Results ---
    print(f"\n{'='*60}")
    print(f"RESULTS: {CIRCUIT_SIZE}q QNN on {BACKEND_NAME}")
    print(f"{'='*60}")
    print(f"Compile time:        {compile_time:.1f}s")
    print(f"Execution time:      {run_time:.1f}s")
    print(f"Estimated QPU time:  {estimated_qpu_time:.4f}s")
    print(f"Actual QPU time:     {backend.total_actual_qpu_time}s")
    print(f"Total jobs:          {backend.total_jobs}")

    if backend.total_actual_qpu_time > 0:
        ratio = backend.total_actual_qpu_time / estimated_qpu_time
        print(f"Actual/Estimated:    {ratio:.2f}x")

    print(f"\nReconstructed result shape: {result.shape}")
    print(f"Result value: {result.item() if result.ndim == 0 else result}")

    # --- Compare with noiseless simulation ---
    print("\nRunning noiseless simulation for comparison...")
    from qtpu.runtime.backends import CudaQBackend

    sim_backend = CudaQBackend(target="qpp-cpu", simulate=True, estimate_qpu_time=False)
    sim_runtime = HEinsumRuntime(htn, backend=sim_backend, dtype=torch.float64)
    sim_runtime.prepare(optimize=False)
    sim_result, sim_timing = sim_runtime.execute()

    print(f"Noiseless result: {sim_result.item() if sim_result.ndim == 0 else sim_result}")
    print(f"Hardware result:  {result.item() if result.ndim == 0 else result}")

    if result.ndim == 0 and sim_result.ndim == 0:
        error = abs(result.item() - sim_result.item())
        print(f"Absolute error:   {error:.6f}")


if __name__ == "__main__":
    main()
