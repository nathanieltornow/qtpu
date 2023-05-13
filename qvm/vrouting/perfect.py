from qiskit.circuit import QuantumCircuit, Barrier
from qiskit.transpiler import CouplingMap

from qvm.virtual_gates import VIRTUAL_GATE_TYPES


def vroute_perfect(
    circuit: QuantumCircuit, coupling_map: CouplingMap, initial_layout: list[int]
) -> QuantumCircuit:
    """Route a circuit perfectly using gate virtualization.

    Args:
        circuit: The circuit to route.
        coupling_map: The coupling map of the target device.
        initial_layout: The initial layout of the circuit.

    Returns:
        The routed circuit.
    """
    assert circuit.num_qubits == len(
        initial_layout
    ), "Initial layout does not match circuit size."

    res_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)

    qubit_to_physical = {
        qubit: initial_layout[i] for i, qubit in enumerate(circuit.qubits)
    }
    for instr in circuit:
        op, qubits, clbits = instr.operation, instr.qubits, instr.clbits
        if len(qubits) > 2 and not isinstance(op, Barrier):
            raise NotImplementedError(
                "Virtualization of multi-qubit gates is not supported."
            )
        elif len(qubits) == 2:
            q0, q1 = qubits
            p0, p1 = qubit_to_physical[q0], qubit_to_physical[q1]
            if coupling_map.distance(p0, p1) > 1:
                op = VIRTUAL_GATE_TYPES[op.name](op)
        res_circuit.append(op, qubits, clbits)
    return res_circuit
