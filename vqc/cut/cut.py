from abc import ABC, abstractmethod
from typing import Dict, Type

from qiskit.circuit import Qubit
from qiskit.dagcircuit import DAGCircuit

from vqc.virtual_gate import VirtualBinaryGate, VirtualCZ, VirtualCX, VirtualRZZ

STANDARD_VIRTUAL_GATES: Dict[str, Type[VirtualBinaryGate]] = {
    "cz": VirtualCZ,
    "cx": VirtualCX,
    "rzz": VirtualRZZ,
}


def cut_qubit_connection(
    dag: DAGCircuit,
    qarg1: Qubit,
    qarg2: Qubit,
    vgate_type: Dict[str, Type[VirtualBinaryGate]] = STANDARD_VIRTUAL_GATES,
) -> None:
    """
    Cut the connection between two qubits in a DAGCircuit by replacing binary gates
    between the two qubits with a corresponding virtual gate.

    Args:
        dag (DAGCircuit): The DAGCircuit to cut.
        qarg1 (Qubit): The first qubit.
        qarg2 (Qubit): The second qubit.
        vgate_type (Dict[str, Type[VirtualBinaryGate]], optional): Gate-names to the types
            of virtual gate to use for the type of gates with that name.
            Defaults to STANDARD_VIRTUAL_GATES.
    """

    for op_node in dag.op_nodes():
        if op_node.name == "barrier":
            continue
        if (
            len(op_node.qargs) == 2
            and len(op_node.cargs) == 0
            and set(op_node.qargs) <= {qarg1, qarg2}
        ):
            dag.substitute_node(
                op_node, vgate_type[op_node.op.name](op_node.op.params), inplace=True
            )
