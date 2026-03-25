"""Circuit cutting with qTPU.

Cuts a 20-qubit distributed VQE circuit into 10-qubit subcircuits,
converts to a hybrid tensor network (HEinsum), and executes using
the CUDA-Q backend.

Usage:
    uv run python examples/cutting.py
"""

from time import perf_counter

import qtpu
from qtpu import HEinsumRuntime
from evaluation.benchmarks import get_benchmark


def main():
    # -- 1. Generate a 20-qubit distributed VQE benchmark circuit -----------
    circuit = get_benchmark("dist-vqe", circuit_size=20, cluster_size=10)
    print(f"Original circuit: {circuit.num_qubits} qubits, {circuit.size()} gates")

    # -- 2. Cut the circuit into subcircuits of at most 10 qubits -----------
    t0 = perf_counter()
    cut_circuit = qtpu.cut(circuit, max_size=10)
    cut_time = perf_counter() - t0
    print(f"Cutting took {cut_time:.2f}s")

    # -- 3. Convert the cut circuit into an HEinsum specification -----------
    #    This creates qTensors (one per subcircuit) and cTensors (QPD
    #    coefficient tensors for each cut).
    heinsum = qtpu.circuit_to_heinsum(cut_circuit)
    print(
        f"HEinsum: {len(heinsum.quantum_tensors)} qTensors, "
        f"{len(heinsum.classical_tensors)} cTensors"
    )
    print(f"Einsum expression: {heinsum.einsum_expr}")

    for i, qt in enumerate(heinsum.quantum_tensors):
        print(f"  qTensor[{i}]: {qt.circuit.num_qubits} qubits, shape {qt.shape}")

    # -- 4. Prepare the runtime (optimize contraction path, compile circuits)
    runtime = HEinsumRuntime(heinsum, backend="cudaq")
    runtime.prepare()

    # -- 5. Execute the hybrid tensor network contraction -------------------
    result, timing = runtime.execute()
    print(f"\nResult: {result}")

    # -- 6. Timing breakdown ------------------------------------------------
    print(f"\nTiming breakdown:")
    print(f"  Quantum evaluation : {timing.quantum_eval_time:.4f}s")
    print(f"  Classical contract : {timing.classical_contraction_time:.4f}s")
    print(f"  Total              : {timing.total_time:.4f}s")
    print(f"  Circuits executed  : {timing.num_circuits}")


if __name__ == "__main__":
    main()
