from qiskit.dagcircuit import DAGCircuit

from qvm.cut.cut import CutPass, cut_qubit_connection


class LadderDecomposition(CutPass):
    def __init__(self, num_partitions: int) -> None:
        self.num_partitions = num_partitions

    def run(self, dag: DAGCircuit) -> None:
        num_frags = min(self.num_partitions, dag.num_qubits())
        frag_size = dag.num_qubits() // num_frags
        cut_qubits = [(frag_size * i - 1, frag_size * i) for i in range(1, num_frags)]
        for qubit1_index, qubit2_index in cut_qubits:
            cut_qubit_connection(
                dag, dag.qubits[qubit1_index], dag.qubits[qubit2_index]
            )
