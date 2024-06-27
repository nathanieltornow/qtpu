import quimb.tensor as qtn
from qiskit.circuit import QuantumCircuit, Gate, QuantumRegister, ClassicalRegister


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


def defer_mid_measurements(circuit: QuantumCircuit) -> QuantumCircuit:
    new_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)

    final_measures = []
    qubits = set()
    for i, instr in enumerate(reversed(circuit)):
        if instr.operation.name == "measure" and instr.qubits[0] not in qubits:
            final_measures.append(i)
        qubits.add(instr.qubits[0])
        if len(qubits) == circuit.num_qubits:
            break

    meas_ctr = 0
    for i, instr in enumerate(circuit):
        if (
            instr.operation.name == "measure"
            and len(circuit) - i - 1 not in final_measures
        ):
            new_qr = QuantumRegister(1, name=f"defer_{meas_ctr}")
            meas_ctr += 1
            new_circuit.add_register(new_qr)
            new_circuit.cx(instr.qubits[0], new_qr)
            new_circuit.measure(new_qr, instr.clbits[0])
            continue

        if instr.operation.name != "reset":
            new_circuit.append(instr)

    return merge_regs(new_circuit)


def merge_regs(circuit: QuantumCircuit) -> QuantumCircuit:
    qreg = QuantumRegister(circuit.num_qubits, name="q")
    creg = ClassicalRegister(circuit.num_clbits, name="c")
    new_circuit = QuantumCircuit(qreg, creg)
    for instr in circuit:
        new_circuit.append(
            instr.operation,
            [qreg[circuit.qubits.index(q)] for q in instr.qubits],
            [creg[circuit.clbits.index(c)] for c in instr.clbits],
        )
    return new_circuit
