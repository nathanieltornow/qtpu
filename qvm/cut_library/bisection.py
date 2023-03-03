from networkx.algorithms.community import kernighan_lin_bisection
from qiskit.circuit import QuantumCircuit, Qubit

from qvm.virtual_gates import WireCut
from qvm.cut_library.util import circuit_to_qcg, decompose_qubits, circuit_to_dag


def bisect_gate(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Decomposes a circuit into two fragments through gate virtualization
    using the Kernighan-Lin Bisection of the qubit connectivity graph.

    Args:
        circuit (QuantumCircuit): The circuit.

    Returns:
        QuantumCircuit: The bisected circuit.
    """
    qcg = circuit_to_qcg(circuit)
    A, B = kernighan_lin_bisection(qcg)
    return decompose_qubits(circuit, [A, B])


def bisect_wire(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Decomposes a circuit into two fragments through gate virtualization
    using the Kernighan-Lin Bisection of the DAG.

    Args:
        circuit (QuantumCircuit): The circuit.

    Returns:
        QuantumCircuit: The bisected circuit.
    """
    raise NotImplementedError("Bug")
    dag = circuit_to_dag(circuit)
    A, B = kernighan_lin_bisection(dag.to_undirected())
    bisected_circuit = QuantumCircuit(
        *circuit.qregs,
        *circuit.cregs,
        name=circuit.name,
        global_phase=circuit.global_phase,
        metadata=circuit.metadata,
    )

    wires_to_cut: set[tuple[int, int]] = set(
        (int(u), int(v)) for u, v in dag.edges if u in A and v in B or u in B and v in A
    )

    def _next_operation_on_qubit(from_index: int, qubit: Qubit) -> int:
        for i, cinstr in enumerate(circuit.data[from_index + 1 :]):
            if qubit in cinstr.qubits:
                return i + from_index + 1
        return -1

    # op_done_on_qubit: dict[Qubit, bool] = {qubit: False for qubit in circuit.qubits}
    for i, cinstr in enumerate(circuit.data):
        for qubit in cinstr.qubits:
            j = _next_operation_on_qubit(i, qubit)
            if (i, j) in wires_to_cut or (j, i) in wires_to_cut:
                bisected_circuit.append(WireCut(), [qubit], [])
        bisected_circuit.append(cinstr, cinstr.qubits, cinstr.clbits)
    return bisected_circuit
