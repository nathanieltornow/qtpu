from networkx.algorithms.community import kernighan_lin_bisection
from qiskit.circuit import QuantumCircuit

from qvm.cut_library._cut import CutStrategy
from qvm.cut_library.util import circuit_to_qcg, decompose_qubits


class GateBisection(CutStrategy):
    """
    Decomposes a circuit into two fragments through 
    Kernighan-Lin Bisection of the qubit connectivity graph.
    """

    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        qcg = circuit_to_qcg(circuit)
        A, B = kernighan_lin_bisection(qcg)
        return decompose_qubits(circuit, [A, B])
