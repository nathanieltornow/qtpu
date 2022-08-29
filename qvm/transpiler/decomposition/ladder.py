from typing import Optional
from qiskit.circuit import QuantumCircuit
from qiskit.providers import Backend

from qvm.transpiler.transpiler import VirtualTranspiler, virtualize_connection


class LadderDecomposition(VirtualTranspiler):
    def __init__(self, num_partitions: int) -> None:
        self.num_partitions = num_partitions

    def run(
        self, circuit: QuantumCircuit, backend: Optional[Backend] = None
    ) -> QuantumCircuit:
        num_frags = min(self.num_partitions, circuit.num_qubits)
        frag_size = circuit.num_qubits // num_frags
        cut_qubits = [(frag_size * i - 1, frag_size * i) for i in range(1, num_frags)]
        for qubit1, qubit2 in cut_qubits:
            virtualize_connection(circuit, qubit1, qubit2)
        return circuit
