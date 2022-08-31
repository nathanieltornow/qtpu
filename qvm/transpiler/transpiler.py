from abc import ABC, abstractmethod
from typing import Dict, Type

from qiskit.circuit import Qubit
from qiskit.transpiler.basepasses import BasePass
from qiskit.dagcircuit import DAGCircuit

from qvm.virtual_gate import VirtualBinaryGate, VirtualCZ, VirtualCX, VirtualRZZ
from .fragmented_circuit import FragmentedCircuit

STANDARD_VIRTUAL_GATES: Dict[str, Type[VirtualBinaryGate]] = {
    "cz": VirtualCZ,
    "cx": VirtualCX,
    "rzz": VirtualRZZ,
}


class VirtualizationPass(BasePass):
    @abstractmethod
    def run(self, dag: DAGCircuit) -> None:
        pass


class DistributedPass(ABC):
    @abstractmethod
    def run(self, frag_circ: FragmentedCircuit) -> None:
        pass


def virtualize_connection(
    dag: DAGCircuit,
    qarg1: Qubit,
    qarg2: Qubit,
    vgate_type: Dict[str, Type[VirtualBinaryGate]] = STANDARD_VIRTUAL_GATES,
) -> None:
    for op_node in dag.op_nodes:
        if (
            len(op_node.qargs) == 2
            and len(op_node.qargs) == 0
            and set(op_node.qargs) <= {qarg1, qarg2}
        ):
            dag.substitute_node(
                op_node, vgate_type[op_node.name](op_node.params), inplace=True
            )
