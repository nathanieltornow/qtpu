import itertools
from typing import List, Set
from qiskit.circuit import Qubit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass

from vqc.cut.cut import cut_qubit_connection
from vqc.converters import dag_to_connectivity_graph


class QubitGroups(TransformationPass):
    def __init__(self, groups: List[Set[Qubit]]):
        assert len(groups) > 0
        self.groups = groups
        super().__init__()

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        for group1, group2 in itertools.combinations(self.groups, 2):
            for qubit1, qubit2 in itertools.product(group1, group2):
                cut_qubit_connection(dag, qubit1, qubit2)
