from typing import Optional
from networkx.algorithms.community import kernighan_lin_bisection
from qiskit.circuit import QuantumCircuit
from qiskit.providers import Backend

from qvm.transpiler.transpiler import VirtualTranspiler, virtualize_connection


class Bisection(VirtualTranspiler):
    def run(
        self, circuit: QuantumCircuit, backend: Optional[Backend] = None
    ) -> QuantumCircuit:
        A, B = kernighan_lin_bisection(circuit.graph)
        for nodeA in A:
            for nodeB in B:
                virtualize_connection(circuit, nodeA, nodeB)
        return circuit
