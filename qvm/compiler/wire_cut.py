from qiskit.circuit import CircuitInstruction

from qvm.dag import DAG
from qvm.virtual_gates import WireCut

from ._asp import dag_to_asp, get_optimal_symbols


def cut_wires(dag: DAG, size_to_reach: int) -> dict[int, int]:
    from clingo.control import Control

    min_num_fragments = len(dag.qubits) // size_to_reach
    partitions: dict[int, int] | None = None
    while partitions is None:
        partitions = _find_optimal_partitons(dag, min_num_fragments, size_to_reach)
        min_num_fragments += 1

    edges = list(dag.edges)
    for u, v in edges:
        if partitions[u] != partitions[v]:
            dag.remove_edge(u, v)

            qubits = set(dag.get_node_instr(u).qubits) & set(
                dag.get_node_instr(v).qubits
            )
            for qubit in qubits:
                new_instr = CircuitInstruction(WireCut(), [qubit])
                w = dag.add_instr_node(new_instr)
                dag.add_edge(u, w)
                dag.add_edge(w, v)

    return partitions


def _find_optimal_partitons(
    dag: DAG, num_fragments: int, size_to_reach: int
) -> dict[int, int] | None:
    from clingo.control import Control

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
    """
    return asp
