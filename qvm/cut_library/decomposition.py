from networkx.algorithms.community import kernighan_lin_bisection
from qiskit.circuit import QuantumCircuit, Qubit

from qvm.cut_library.util import circuit_to_qcg, decompose_qubits


def bisect(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Decomposes a circuit into two fragments through gate virtualization
    using the Kernighan-Lin Bisection of the qubit connectivity graph.

    Args:
        circuit (QuantumCircuit): The circuit.

    Returns:
        QuantumCircuit: The bisected circuit.
    """
    qcg = circuit_to_qcg(circuit)
    A, B = kernighan_lin_bisection(qcg)
    return decompose_qubits(circuit, [A, B])


def bisect_recursive(circuit: QuantumCircuit, num_fragments: int) -> QuantumCircuit:
    """
    Decomposes a circuit into a given number of fragments by recursively bisecting the circuit.
    On each step, the largest fragment is bisected.

    Args:
        circuit (QuantumCircuit): The circuit.
        num_fragments (int): The number of fragments to decompose the circuit into.

    Returns:
        QuantumCircuit: The decomposed circuit.
    """
    qcg = circuit_to_qcg(circuit)
    fragment_qubits: list[set[Qubit]]
    fragment_qubits = list(kernighan_lin_bisection(qcg))
    for _ in range(num_fragments - 2):
        largest_fragment = max(fragment_qubits, key=lambda f: len(f))
        fragment_qubits.remove(largest_fragment)
        fragment_qubits += list(kernighan_lin_bisection(qcg.subgraph(largest_fragment)))
    print([len(fragment) for fragment in fragment_qubits])
    return decompose_qubits(circuit, fragment_qubits)
