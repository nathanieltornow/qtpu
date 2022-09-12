from networkx.algorithms.community import kernighan_lin_bisection
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass

from vqc.cut.cut import cut_qubit_connection
from vqc.converters import dag_to_connectivity_graph


class Bisection(TransformationPass):
    def run(self, dag: DAGCircuit) -> DAGCircuit:
        cg = dag_to_connectivity_graph(dag)
        A, B = kernighan_lin_bisection(cg)
        for nodeA in A:
            for nodeB in B:
                cut_qubit_connection(dag, nodeA, nodeB)
        return dag
