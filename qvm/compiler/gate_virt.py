from networkx.algorithms.community import kernighan_lin_bisection
from qiskit.circuit import Qubit

from qvm.virtual_gates import VIRTUAL_GATE_TYPES
from qvm.dag import DAG

from .util import dag_to_qcg
from ._asp import dag_to_asp, qcg_to_asp, get_optimal_symbols


def decompose_qubit_sets(dag: DAG, qubit_sets: list[set[Qubit]]) -> None:
    for node in dag.nodes:
        instr = dag.get_node_instr(node)
        qubits = instr.qubits

        nums_of_frags = sum(1 for qubit_set in qubit_sets if set(qubits) & qubit_set)
        if nums_of_frags == 0:
            raise ValueError(f"No fragment found for qubit {qubits}.")
        elif nums_of_frags > 1:
            instr.operation = VIRTUAL_GATE_TYPES[instr.operation.name](instr.operation)


def bisect_recursive(dag: DAG, size_to_reach: int) -> None:
    qcg = dag_to_qcg(dag)
    fragment_qubits: list[set[Qubit]]
    fragment_qubits = list(kernighan_lin_bisection(qcg))

    while any(len(f) > size_to_reach for f in fragment_qubits):
        largest_fragment = max(fragment_qubits, key=lambda f: len(f))
        fragment_qubits.remove(largest_fragment)
        fragment_qubits += list(kernighan_lin_bisection(qcg.subgraph(largest_fragment)))

    decompose_qubit_sets(dag, fragment_qubits)


def optimal_gate_cut(dag: DAG, size_to_reach: int) -> None:
    qcg = dag_to_qcg(dag, use_qubit_idx=True)
    asp = qcg_to_asp(qcg)
    num_partitions = len(dag.qubits) // size_to_reach + (len(dag.qubits) % size_to_reach != 0)
    asp += _gate_cut_asp(num_partitions=num_partitions)

    symbols = get_optimal_symbols(asp)

    qubit_sets: list[set[Qubit]] = [set() for _ in range(num_partitions)]
    for symbol in symbols:
        if symbol.name != "qubit_in_partition":
            continue
        qubit_idx = symbol.arguments[0].number
        partition_idx = symbol.arguments[1].number
        qubit_sets[partition_idx].add(dag.qubits[qubit_idx])

    decompose_qubit_sets(dag, qubit_sets)


def minimize_qubit_dependencies(dag: DAG, max_virt: int) -> None:
    asp = dag_to_asp(dag)
    asp += _min_dep_asp(max_virt)

    symbols = get_optimal_symbols(asp)

    virtualized_nodes: set[int] = set()
    for symbol in symbols:
        if symbol.name != "vgate":
            continue
        gate_idx = symbol.arguments[0].number
        virtualized_nodes.add(gate_idx)

    for vnode in virtualized_nodes:
        instr = dag.get_node_instr(vnode)
        instr.operation = VIRTUAL_GATE_TYPES[instr.operation.name](instr.operation)


def _min_dep_asp(max_virt: int) -> str:
    asp = f"""
    {{ vgate(Gate) : gate(Gate) }}.
    
    :- N = #count{{Gate : vgate(Gate)}}, N > {max_virt}.
    
    path(Gate1, Gate2) :- wire(_, Gate1, Gate2), not vgate(Gate1).
    path(Gate1, Gate3) :- path(Gate1, Gate2), path(Gate2, Gate3).
    
    depends_on(Qubit1, Qubit2) :- 
        gate_on_qubit(Gate1, Qubit1),
        gate_on_qubit(Gate2, Qubit2),
        path(Gate1, Gate2),
        Qubit1 != Qubit2.
    
    num_deps(N) :- N = #count{{Qubit1, Qubit2 : depends_on(Qubit1, Qubit2)}}.
    
    num_vgates(N) :- N = #count{{Gate : vgate(Gate)}}.
    
    #minimize{{N : num_deps(N)}}.
    #minimize{{N : num_vgates(N)}}.
    #show vgate/1.
    """
    return asp


def _gate_cut_asp(num_partitions: int) -> str:
    asp = f"""
    partition(P) :- P = 0..{num_partitions - 1}.
    
    {{ qubit_in_partition(Qubit, P) : partition(P) }} == 1 :- qubit(Qubit).
    :- partition(P), not qubit_in_partition(_, P).
    
    partition_pair(P1, P2) :- partition(P1), partition(P2), P1 < P2.

    qubit_conn_between_partitions(Qubit1, Qubit2, W) :- 
        qubit(Qubit1),
        qubit(Qubit2),
        qubit_in_partition(Qubit1, P1),
        qubit_in_partition(Qubit2, P2),
        partition_pair(P1, P2),
        qubit_conn(Qubit1, Qubit2, W).

    num_vgates(N) :- N = #sum{{ W, Qubit1, Qubit2 : qubit_conn_between_partitions(Qubit1, Qubit2, W) }}.
    
    #minimize{{ N : num_vgates(N) }}.
    #show qubit_in_partition/2.
    """
    return asp