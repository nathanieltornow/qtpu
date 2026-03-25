"""Simple circuit cutting and execution with qTPU.

Cuts a small circuit in half, converts to an HEinsum, and
contracts it using the CUDA-Q backend.

Usage:
    uv run python examples/sampling.py
"""

from qiskit.circuit import QuantumCircuit

import qtpu
from qtpu import HEinsumRuntime


def main():
    # Build a 6-qubit circuit with entanglement across the cut boundary
    qc = QuantumCircuit(6, 6)
    for i in range(6):
        qc.h(i)
    for i in range(5):
        qc.cx(i, i + 1)
    qc.measure(range(6), range(6))

    print(f"Circuit: {qc.num_qubits} qubits, {qc.size()} gates")

    # Cut into subcircuits of at most 3 qubits
    cut_circuit = qtpu.cut(qc, max_size=3)
    heinsum = qtpu.circuit_to_heinsum(cut_circuit)

    print(
        f"HEinsum: {len(heinsum.quantum_tensors)} qTensors, "
        f"{len(heinsum.classical_tensors)} cTensors"
    )

    # Execute
    runtime = HEinsumRuntime(heinsum, backend="cudaq")
    runtime.prepare()
    result, timing = runtime.execute()

    print(f"Result: {result}")
    print(f"Total time: {timing.total_time:.3f}s")


if __name__ == "__main__":
    main()
