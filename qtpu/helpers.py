import quimb.tensor as qtn
from qiskit.circuit import QuantumCircuit, Gate


def qiskit_to_quimb(circuit: QuantumCircuit) -> qtn.Circuit:
    circ = qtn.Circuit(circuit.num_qubits)
    for instr in circuit:
        op, qubits = instr.operation, instr.qubits
        assert isinstance(op, Gate)

        circ.apply_gate_raw(op.to_matrix(), [circuit.qubits.index(q) for q in qubits])

    return circ


def sample_quimb(circuit: QuantumCircuit, shots: int) -> dict[str, int]:
    circuit.remove_final_measurements()

    tn_circ = qiskit_to_quimb(circuit)

    counts = {}
    for sample in tn_circ.sample(shots):
        sample_str = "".join(reversed(sample))
        counts[sample_str] = counts.get(sample_str, 0) + 1
    return counts


def compute_Z_expectation(circuit: QuantumCircuit) -> float:
    from cuquantum import CircuitToEinsum, contract
    import cupy as cp

    myconverter = CircuitToEinsum(circuit, dtype="complex128", backend=cp)
    pauli_string = "Z" * circuit.num_qubits
    expression, operands = myconverter.expectation(pauli_string, lightcone=True)
    return contract(expression, *operands)
