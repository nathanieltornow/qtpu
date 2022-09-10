from qiskit.circuit import Qubit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass

from qvm.cut.cut import cut_qubit_connection


class StaticCut(TransformationPass):
    def __init__(self, qubit1: Qubit, qubit2: Qubit):
        self.qubit1 = qubit1
        self.qubit2 = qubit2

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        cut_qubit_connection(dag, self.qubit1, self.qubit2)
        return dag
