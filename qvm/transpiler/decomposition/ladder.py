from qiskit.dagcircuit import DAGCircuit

from qvm.transpiler.transpiler import VirtualizationPass, virtualize_connection


class LadderDecomposition(VirtualizationPass):
    def __init__(self, num_partitions: int) -> None:
        self.num_partitions = num_partitions

    def run(self, dag: DAGCircuit) -> None:
        num_frags = min(self.num_partitions, dag.num_qubits)
        frag_size = dag.num_qubits // num_frags
        cut_qubits = [(frag_size * i - 1, frag_size * i) for i in range(1, num_frags)]
        for qubit1, qubit2 in cut_qubits:
            virtualize_connection(dag, qubit1, qubit2)
