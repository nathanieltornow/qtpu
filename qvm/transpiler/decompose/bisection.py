from networkx.algorithms.community import kernighan_lin_bisection

from qvm.circuit.virtual_circuit import VirtualCircuitInterface
from qvm.transpiler.transpiler import QVMTranspiler


class Bisection(QVMTranspiler):
    def run(self, circuit: VirtualCircuitInterface) -> None:
        A, B = kernighan_lin_bisection(circuit.graph)
        for nodeA in A:
            for nodeB in B:
                circuit.virtualize_connection(nodeA, nodeB)
