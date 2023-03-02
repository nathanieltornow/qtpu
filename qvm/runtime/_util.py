from qiskit.circuit import QuantumCircuit, Parameter


def circuit_hash(circuit: QuantumCircuit) -> int:
    return hash(
        tuple(
            (instr.name, *instr.qubits, *instr.clbits, *instr.params)
            for instr in circuit.data
        )
    )
