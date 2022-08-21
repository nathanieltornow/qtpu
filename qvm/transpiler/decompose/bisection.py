from networkx.algorithms.community import kernighan_lin_bisection

from qvm.circuit import FragmentedVirtualCircuit
from qvm.transpiler.transpiler import DecompositionTranspiler


class Bisection(DecompositionTranspiler):
    def run(self, circuit: FragmentedVirtualCircuit) -> None:
        """
        Decompose a fragmented circuit into fragments using the kernighan_lin_bisection algorithm.
        """
        # merge all fragments
        circuit.merge_fragments(circuit.fragments())
        # determine the node sets using the kernighan_lin_bisection algorithm
        A, B = kernighan_lin_bisection(circuit.connectivity_graph())
        for nodeA in A:
            for nodeB in B:
                circuit.virtualize_connection(nodeA, nodeB)
        circuit.create_fragments()
