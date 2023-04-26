import itertools

from networkx.algorithms.community import kernighan_lin_bisection
from qiskit.circuit import Barrier, QuantumCircuit, Qubit

from qvm.converters import circuit_to_qcg, fragment_circuit
from qvm.virtual_gates import VIRTUAL_GATE_TYPES


def decompose_qubits(
    circuit: QuantumCircuit, con_qubits: list[set[Qubit]]
) -> QuantumCircuit:
    """
    Decomposes a circuit using gate virtualization.
    The fragments are defined by the connected qubits, which should still be connected.

    Args:
        circuit (QuantumCircuit): The original circuit.
        con_qubits (list[set[Qubit]]): The connected qubits.
            Each set of qubits is a fragment.
            The qubit set need to be disjoint and contain all qubits of the circuit.

    Raises:
        ValueError: Thrown if con_qubits is illegal.

    Returns:
        QuantumCircuit: The decomposed circuit with virtual gates.
    """
    if set(circuit.qubits) != set.union(*con_qubits):
        raise ValueError("con_qubits is not containing all qubits of the circuit.")
    if len(list(itertools.chain(*con_qubits))) != len(circuit.qubits):
        raise ValueError("con_qubits is not disjoint.")

    def _in_multiple_fragments(qubits: set[Qubit]) -> bool:
        for qubit_set in con_qubits:
            if qubit_set & qubits and not qubits <= qubit_set:
                return True
            if qubits <= qubit_set:
                return False
        return False

    new_circ = QuantumCircuit(
        *circuit.qregs,
        *circuit.cregs,
        name=circuit.name,
        global_phase=circuit.global_phase,
        metadata=circuit.metadata,
    )
    for cinstr in circuit.data:
        op, qubits, clbits = cinstr.operation, cinstr.qubits, cinstr.clbits
        if _in_multiple_fragments(set(qubits)) and not isinstance(op, Barrier):
            op = VIRTUAL_GATE_TYPES[op.name](op)
        new_circ.append(op, qubits, clbits)
    return fragment_circuit(new_circ)


def bisect(circuit: QuantumCircuit, num_fragments: int = 2) -> QuantumCircuit:
    """
    Decomposes a circuit into a given number of fragments by recursively bisecting the circuit.
    On each step, the largest fragment is bisected.

    Args:
        circuit (QuantumCircuit): The circuit.
        num_fragments (int): The number of fragments to decompose the circuit into.

    Returns:
        QuantumCircuit: The decomposed circuit.
    """
    qcg = circuit_to_qcg(circuit)
    fragment_qubits: list[set[Qubit]]
    fragment_qubits = list(kernighan_lin_bisection(qcg))
    for _ in range(num_fragments - 2):
        largest_fragment = max(fragment_qubits, key=lambda f: len(f))
        fragment_qubits.remove(largest_fragment)
        fragment_qubits += list(kernighan_lin_bisection(qcg.subgraph(largest_fragment)))
    return decompose_qubits(circuit, fragment_qubits)


def cut_gates_optimal(
    circuit: QuantumCircuit,
    num_fragments: int = 2,
    max_cuts: int | None = None,
    max_fragment_size: int | None = None,
) -> QuantumCircuit:
    pass
