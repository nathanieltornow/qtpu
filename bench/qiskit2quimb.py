import quimb.tensor as qtn
from qiskit.circuit import QuantumCircuit, Gate


def qiskit_to_quimb(circuit: QuantumCircuit) -> qtn.Circuit:
    circ = qtn.Circuit(circuit.num_qubits)
    for instr in circuit:
        op, qubits = instr.operation, instr.qubits
        assert isinstance(op, Gate)

        circ.apply_gate_raw(op.to_matrix(), [circuit.qubits.index(q) for q in qubits])

    return circ


def sample_with_quimb(circuit: QuantumCircuit, shots: int) -> dict[str, int]:
    circuit.remove_final_measurements()

    tn_circ = qiskit_to_quimb(circuit)

    counts = {}
    for sample in tn_circ.sample(shots):
        sample_str = "".join(reversed(sample))
        counts[sample_str] = counts.get(sample_str, 0) + 1
    return counts


if __name__ == "__main__":
    N = 100
    circuit = QuantumCircuit(N)
    circuit.h(0)
    circuit.cx(range(N-1), range(1, N))
    # circuit.rzz(2.0, 1, 2)

    from qiskit_aer import AerSimulator

    # tn = qiskit_to_quimb(circuit)

    # for sample in tn.sample(1000):
    #     print(sample)

    circuit.measure_all()
    print(AerSimulator(method="matrix_product_state").run(circuit, shots=1000).result().get_counts())
    print(sample_with_quimb(circuit, 1000))

    # tn:qtn.TensorNetwork = tn.psi

    # fig = tn.draw(return_fig=True)
    # fig.savefig("qiskit2quimb.png")
