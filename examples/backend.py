from collections import Counter


from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import hellinger_fidelity
from qiskit_aer import AerSimulator

import qtpu
from qtpu.evaluators import BackendEvaluator

from _helper import simple_circuit


def sample_circuit_qtpu(circuit: QuantumCircuit, num_shots: int) -> dict[str, int]:
    """
    Samples the given circuit using QTPU.
    Returns the counts of the circuit.
    """
    # cut the circuit into two halves
    cut_circ = qtpu.cut(circuit, num_qubits=circuit.num_qubits // 2)

    # convert the circuit into a hybrid tensor network
    hybrid_tn = qtpu.circuit_to_hybrid_tn(cut_circ)

    # evaluate the hybrid tensor network to a classical tensor network
    # using sampling on a backend
    backend = AerSimulator()
    evaluator = BackendEvaluator(backend=backend, individual_jobs=True)
    tn = qtpu.evaluate(hybrid_tn, evaluator)

    # this now gives us a classical tensor network which represents the
    # probability distribution of the circuit

    # from this tensor network, we can now sample
    sample_results = qtpu.sample(tn, num_samples=num_shots)
    return dict(Counter(sample_results))


def run_comparison(circuit: QuantumCircuit, num_shots: int) -> dict[str, int]:
    counts = AerSimulator().run(circuit, shots=num_shots).result().get_counts()
    return counts


def main():
    circuit = simple_circuit(4)
    qtpu_counts = sample_circuit_qtpu(circuit, 10000)
    qiskit_counts = run_comparison(circuit, 10000)
    print(f"QTPU counts: {qtpu_counts}")
    print(f"Qiskit counts: {qiskit_counts}")
    print(f"Hellinger Fidelity: {hellinger_fidelity(qtpu_counts, qiskit_counts)}")


if __name__ == "__main__":
    main()
