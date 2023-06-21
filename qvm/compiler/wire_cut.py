import networkx as nx
from qiskit.circuit import CircuitInstruction, Qubit, QuantumRegister

from qvm.dag import DAG
from qvm.virtual_gates import WireCut

from ._asp import dag_to_asp, get_optimal_symbols


def cut_wires(dag: DAG, size_to_reach: int) -> None:
    min_num_fragments = len(dag.qubits) // size_to_reach
    partitions: dict[int, int] | None = None
    while partitions is None:
        partitions = _find_optimal_partitons(dag, min_num_fragments, size_to_reach)
        min_num_fragments += 1

    edges = list(dag.edges())
    i = 0
    for u, v in edges:
        if partitions[u] != partitions[v]:
            i += 1
            dag.remove_edge(u, v)
            qubits = set(dag.get_node_instr(u).qubits) & set(
                dag.get_node_instr(v).qubits
            )
            for qubit in qubits:
                new_instr = CircuitInstruction(WireCut(), [qubit])
                w = dag.add_instr_node(new_instr)
                dag.add_edge(u, w)
                dag.add_edge(w, v)


def wire_cuts_to_vswaps(dag: DAG) -> None:
    raise NotImplementedError()
    qubit_map: dict[Qubit, Qubit] = {}

    def _find_in_map(qubit: Qubit) -> Qubit:
        while qubit in qubit_map:
            qubit = qubit_map[qubit]
        return qubit

    num_wire_cuts = sum(
        1
        for node in dag.nodes
        if isinstance(dag.get_node_instr(node).operation, WireCut)
    )

    for node in nx.topological_sort(dag):
        instr = dag.get_node_instr(node)
        op, qubits = instr.operation, instr.qubits
        qubits = [_find_in_map(qubit) for qubit in qubits]
        if isinstance(op, WireCut):
            pass
        instr.qubits = qubits


def _find_optimal_partitons(
    dag: DAG, num_fragments: int, size_to_reach: int
) -> dict[int, int] | None:
    asp = dag_to_asp(dag)
    asp += _wire_cut_asp(num_fragments=num_fragments, size_to_reach=size_to_reach)

    try:
        symbols = get_optimal_symbols(asp)
    except ValueError:
        return None

    partitions: dict[int, int] = {}
    for symbol in symbols:
        if symbol.name != "gate_in_partition":
            continue
        gate_idx = symbol.arguments[0].number
        partition_idx = symbol.arguments[1].number
        partitions[gate_idx] = partition_idx
    return partitions


def _wire_cut_asp(num_fragments: int, size_to_reach: int) -> str:
    asp = f"""
    partition(P) :- P = 0..{num_fragments - 1}.

    {{ gate_in_partition(Gate, P) : partition(P) }} == 1 :- gate(Gate).
    :- partition(P), not gate_in_partition(_, P).
    
    cutted_wire(Qubit, Gate1, Gate2) :- 
        wire(Qubit, Gate1, Gate2), 
        gate_in_partition(Gate1, P1), 
        gate_in_partition(Gate2, P2), 
        P1 != P2.
        
    num_cutted_wires(N) :- N = #count{{Qubit, Gate1, Gate2 : cutted_wire(Qubit, Gate1, Gate2)}}.
    
    qubit_in_partition(Q, P) :- gate_in_partition(Gate, P), gate_on_qubit(Gate, Q).
    num_qubits_in_partition(P, N) :- partition(P), N = #count{{Q : qubit_in_partition(Q, P)}}.
    :- num_qubits_in_partition(P, N), N > {size_to_reach}.
    
    #minimize{{ N : num_cutted_wires(N) }}.
    
    #show gate_in_partition/2.
    #show num_cutted_wires/1.
    """
    return asp
