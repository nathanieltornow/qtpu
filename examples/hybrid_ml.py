"""Hybrid ML: compile a QNN circuit and inspect the resulting hTN.

Shows how qTPU's compiler transforms a quantum neural network circuit
into a hybrid tensor network with quantum and classical components.
"""

import qtpu
from mqt.bench import get_benchmark_indep


def main():
    # Get a 20-qubit QNN benchmark circuit
    qc = get_benchmark_indep("qnn", 20)
    print(f"Original QNN circuit: {qc.num_qubits} qubits, {qc.size()} gates")

    # Compile: partition into 10-qubit subcircuits
    htn = qtpu.compile_to_heinsum(qc, max_size=10, n_trials=10)

    print(f"\nHybrid tensor network:")
    print(f"  Quantum tensors:  {len(htn.quantum_tensors)}")
    print(f"  Classical tensors: {len(htn.classical_tensors)}")
    print(f"  Output indices:    {htn.output_inds}")

    for i, qt in enumerate(htn.quantum_tensors):
        print(f"  qTensor {i}: indices={qt.inds}, shape={qt.shape}")


if __name__ == "__main__":
    main()
