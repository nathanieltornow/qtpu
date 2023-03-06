from networkx.algorithms.community import kernighan_lin_bisection
from qiskit.circuit import QuantumCircuit

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
