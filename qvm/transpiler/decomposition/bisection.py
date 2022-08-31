from networkx.algorithms.community import kernighan_lin_bisection
from qiskit.dagcircuit import DAGCircuit

from qvm.transpiler.transpiler import VirtualizationPass, virtualize_connection
from qvm.converters import dag_to_connectivity_graph


class Bisection(VirtualizationPass):
    def run(self, dag: DAGCircuit) -> None:
        cg = dag_to_connectivity_graph(dag)
        A, B = kernighan_lin_bisection(cg)
        for nodeA in A:
            for nodeB in B:
                virtualize_connection(dag, nodeA, nodeB)
