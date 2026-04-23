"""Circuit cutting: compile a large circuit into smaller subcircuits."""

import qtpu
from qiskit.circuit.random import random_circuit


def main():
    # Build a 30-qubit circuit
    qc = random_circuit(30, depth=5, seed=42)
    print(f"Original circuit: {qc.num_qubits} qubits")

    # Compile into subcircuits of at most 20 qubits
    htn = qtpu.compile_to_heinsum(qc, max_size=20)

    print(f"Quantum tensors:  {len(htn.quantum_tensors)}")
    print(f"Classical tensors: {len(htn.classical_tensors)}")

    # Inspect the quantum tensors (subcircuits)
    for i, qt in enumerate(htn.quantum_tensors):
        print(f"  qTensor {i}: shape {qt.shape}")


if __name__ == "__main__":
    main()
