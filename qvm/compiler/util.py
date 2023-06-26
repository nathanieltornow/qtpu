from qiskit.circuit import Qubit

from qvm.virtual_gates import VIRTUAL_GATE_TYPES

from .dag import DAG


def decompose_qubit_sets(
    dag: DAG, qubit_sets: list[set[Qubit]], vgate_limit: int = -1
) -> None:
    if vgate_limit == -1:
        vgate_limit = dag.number_of_nodes()
    for node in dag.nodes:
        instr = dag.get_node_instr(node)
        qubits = instr.qubits

        nums_of_frags = sum(1 for qubit_set in qubit_sets if set(qubits) & qubit_set)
        if nums_of_frags == 0:
            raise ValueError(f"No fragment found for qubit {qubits}.")
        elif nums_of_frags > 1 and vgate_limit > 0:
            instr.operation = VIRTUAL_GATE_TYPES[instr.operation.name](instr.operation)
            vgate_limit -= 1
