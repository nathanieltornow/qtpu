from qvm.circuit import VirtualCircuit, virtual_circuit
from qvm.transpiler.transpiler import QVMTranspiler


class LadderDecomposition(QVMTranspiler):
    def __init__(self, num_fragments: int) -> None:
        self.num_fragments = num_fragments

    def run(self, circuit: VirtualCircuit) -> VirtualCircuit:
        num_frags = min(self.num_partitions, circuit.num_qubits)
        frag_size = circuit.num_qubits // num_frags
        cut_qubits = [(frag_size * i - 1, frag_size * i) for i in range(1, num_frags)]
        for q_ind_1, q_ind_2 in cut_qubits:
            circuit.virtualize_connection(
                qubit1=circuit.qubits[q_ind_1], qubit2=circuit.qubits[q_ind_2]
            )
        return circuit
