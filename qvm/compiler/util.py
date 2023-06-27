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
        elif nums_of_frags > 1:
            instr.operation = VIRTUAL_GATE_TYPES[instr.operation.name](instr.operation)
            vgate_limit -= 1
            if vgate_limit == 0:
                return


def virtualize_between_qubit_pairs(
    dag: DAG, qubit_pairs: set[tuple[Qubit, Qubit]], vgate_limit: int = -1
) -> int:
    if vgate_limit == -1:
        vgate_limit = dag.number_of_nodes()
    before = vgate_limit
    for node in dag.nodes:
        instr = dag.get_node_instr(node)
        qubits = instr.qubits
        if len(qubits) == 1:
            continue
        elif len(qubits) > 2:
            raise ValueError(f"Instruction acts on more than two qubits: {instr}")

        elif len(qubits) == 2:
            if (qubits[0], qubits[1]) in qubit_pairs or (
                qubits[1],
                qubits[0],
            ) in qubit_pairs:
                instr.operation = VIRTUAL_GATE_TYPES[instr.operation.name](
                    instr.operation
                )
                vgate_limit -= 1
                if vgate_limit == 0:
                    break
    return before - vgate_limit
