from qiskit.dagcircuit import DAGCircuit

from vqc.cut.cut import cut_qubit_connection, CutPass


class LadderDecomposition(CutPass):
    def __init__(self, num_partitions: int) -> None:
        self.num_partitions = num_partitions
        super().__init__()

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        num_frags = min(self.num_partitions, dag.num_qubits())
        frag_size = dag.num_qubits() // num_frags
        cut_qubits = [(frag_size * i - 1, frag_size * i) for i in range(1, num_frags)]
        for qubit1_index, qubit2_index in cut_qubits:
            cut_qubit_connection(
                dag, dag.qubits[qubit1_index], dag.qubits[qubit2_index]
            )
        return dag
