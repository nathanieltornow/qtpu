import itertools

import networkx as nx
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


def _qcg_to_asp(graph: nx.Graph) -> str:
    asp = ""
    for node, data in graph.nodes(data=True):
        if "weight" not in data:
            asp += f"vertex({node}, 1).\n"
        else:
            asp += f'vertex({node}, {data["weight"]}).\n'
    for u, v, data in graph.edges(data=True):
        if "weight" not in data:
            asp += f"edge({u}, {v}, 1).\n"
        else:
            asp += f'edge({u}, {v}, {data["weight"]}).\n'
    return asp


def cut_gates_optimal(
    circuit: QuantumCircuit,
    num_fragments: int = 2,
    max_cuts: int = 4,
    max_fragment_size: int | None = None,
) -> QuantumCircuit:
    from clingo.control import Control
    import importlib.resources

    asp = _qcg_to_asp(circuit_to_qcg(circuit, use_qubit_idx=True))

    with importlib.resources.path("qvm", "asp") as path:
        asp_file = path / "graph_partition.lp"
        asp += asp_file.read_text()

    asp += f"#const num_partitions = {str(num_fragments)}.\n"

    if max_fragment_size is None:
        max_fragment_size = len(circuit.qubits) // num_fragments + 1

    asp += f":- num_vertices(_, V), V > {max_fragment_size}.\n"

    asp += f":- num_cuts(C), C > {max_cuts}.\n"

    control = Control()
    control.configuration.solve.models = 0  # type: ignore
    control.add("base", [], asp)
    control.ground([("base", [])])
    solve_result = control.solve(yield_=True)  # type: ignore
    opt_model = None
    for model in solve_result:  # type: ignore
        opt_model = model

    if opt_model is None:
        raise ValueError("No solution found.")

    qubits_sets: list[set[Qubit]] = [set() for _ in range(num_fragments)]
    for symbol in opt_model.symbols(shown=True):
        if symbol != "partition" and len(symbol.arguments) != 2:
            continue
        qubit_idx, partition = (
            symbol.arguments[0].number,
            symbol.arguments[1].number,
        )
        qubits_sets[partition].add(circuit.qubits[qubit_idx])

    return decompose_qubits(circuit, qubits_sets)
