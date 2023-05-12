import networkx as nx
from networkx.algorithms.community import kernighan_lin_bisection
from qiskit.circuit import QuantumCircuit, Qubit

from qvm.virtual_gates import VirtualBinaryGate
from qvm.util import circuit_to_qcg, decompose_qubits


def bisect(
    circuit: QuantumCircuit,
    num_fragments: int,
    max_fragment_size: int,
    max_gate_cuts: int,
) -> QuantumCircuit:
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
    res_circuit = decompose_qubits(circuit, fragment_qubits)
    if (
        sum(
            1 for instr in res_circuit if isinstance(instr.operation, VirtualBinaryGate)
        )
        > max_gate_cuts
    ):
        raise ValueError(
            "Couldn't cut the circuit with the given constraints (number of gate cuts)."
        )
    if any(qreg.size > max_fragment_size for qreg in res_circuit.qregs):
        raise ValueError(
            "Couldn't cut the circuit with the given constraints (fragment size)."
        )
    return res_circuit


def cut_gates_optimal(
    circuit: QuantumCircuit,
    num_fragments: int = 2,
    max_wire_cuts: int = 4,
    max_fragment_size: int | None = None,
) -> QuantumCircuit:
    import importlib.resources

    from clingo.control import Control

    asp = _qcg_to_asp(circuit_to_qcg(circuit, use_qubit_idx=True))

    with importlib.resources.path("qvm.core", "asp") as path:
        asp_file = path / "graph_partition.lp"
        asp += asp_file.read_text()

    asp += f"#const num_partitions = {str(num_fragments)}.\n"

    if max_fragment_size is None:
        max_fragment_size = len(circuit.qubits) // num_fragments + 1

    asp += f":- num_vertices(_, V), V > {max_fragment_size}.\n"

    asp += f":- num_cuts(C), C > {max_wire_cuts}.\n"

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
