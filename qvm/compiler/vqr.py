from qiskit.transpiler import CouplingMap

from qvm.virtual_gates import VIRTUAL_GATE_TYPES
from .dag import DAG


def perfect_virtual_qubit_routing(
    dag: DAG, coupling_map: CouplingMap, initial_layout: list[int]
) -> None:
    qubit_map = {dag.qubits[i]: p for i, p in enumerate(initial_layout)}

    for node in dag.nodes:
        instr = dag.get_node_instr(node)
        qubits = instr.qubits
        if len(qubits) > 2:
            raise ValueError("Only 1- and 2-qubit gates are supported.")
        elif len(qubits) == 2:
            dist = coupling_map.distance(qubit_map[qubits[0]], qubit_map[qubits[1]])
            if dist > 1:
                instr.operation = VIRTUAL_GATE_TYPES[instr.operation.name](
                    instr.operation
                )


def apply_virtual_qubit_routing(
    dag: DAG, coupling_map: CouplingMap, initial_layout: list[int]
) -> None:
    raise NotImplementedError
