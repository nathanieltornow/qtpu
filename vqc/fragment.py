from qiskit.circuit import QuantumCircuit, ClassicalRegister, Instruction


class Placeholder(Instruction):
    def __init__(self, name: str):
        super().__init__(name=name, num_qubits=1, num_clbits=1, params=[])


def insert_into_fragment(
    circuit: QuantumCircuit,
    conf_reg_size: int,
    insertion: dict[str, QuantumCircuit],
) -> QuantumCircuit:
    """Inserts instructions (as circuits) into a circuit, replacing placeholders.

    Args:
        circuit: The circuit to insert into.
        insertion: A dictionary mapping placeholder names to QuantumCircuits to insert.

    Returns:
        A circuit with the instructions inserted.
    """
    conf_register = ClassicalRegister(size=conf_reg_size, name="conf")
    new_circuit = QuantumCircuit(
        *circuit.qregs, *circuit.cregs, conf_register, name=circuit.name
    )
    for instruction in circuit.data:
        op, qubits, clbits = (
            instruction.operation,
            instruction.qubits,
            instruction.clbits,
        )
        if isinstance(op, Placeholder):
            if op.key not in insertion:
                continue
            if len(qubits) != 1 and insertion[op.key].num_qubits != 1:
                raise ValueError(
                    f"Placeholder {op.key} and insertion must have one qubit."
                )
            new_circuit.append(
                insertion[op.key].to_instruction(),
                qubits,
                [conf_register[op.meas_index]],
            )
        else:
            new_circuit.append(op, qubits, clbits)
    return new_circuit
