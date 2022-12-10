from networkx.algorithms.community import kernighan_lin_bisection
from qiskit.dagcircuit import DAGCircuit

from vqc.util import dag_to_connectivity_graph
from vqc.cutting.cut import CutPass

from .qubit_groups import QubitGroups


class KernighanLinBisection(CutPass):
    def run(self, dag: DAGCircuit) -> DAGCircuit:
        cg = dag_to_connectivity_graph(dag)
        A, B = kernighan_lin_bisection(cg)
        return QubitGroups([A, B], self.vgates).run(dag)
