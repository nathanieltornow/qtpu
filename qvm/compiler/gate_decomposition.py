from networkx.algorithms.community import kernighan_lin_bisection

from qiskit.circuit import Qubit

from .dag import DAG, dag_to_qcg
from .util import decompose_qubit_sets
from ._asp import qcg_to_asp, get_optimal_symbols


def decompose_qubit_bisection(dag: DAG, size_to_reach: int) -> None:
    qcg = dag_to_qcg(dag)
    partitions: list[set[Qubit]]
    partitions = list(kernighan_lin_bisection(qcg))

    while any(len(f) > size_to_reach for f in partitions):
        largest_fragment = max(partitions, key=lambda f: len(f))
        partitions.remove(largest_fragment)
        partitions += list(kernighan_lin_bisection(qcg.subgraph(largest_fragment)))

    decompose_qubit_sets(dag, partitions)


def decompose_optimal(dag: DAG, size_to_reach: int) -> None:
    qcg = dag_to_qcg(dag, use_qubit_idx=True)
    asp = qcg_to_asp(qcg)
    num_partitions = qcg.number_of_nodes() // size_to_reach + (
        qcg.number_of_nodes() % size_to_reach != 0
    )
    asp += _qcg_partition_asp(
        num_partitions=num_partitions, size_to_reach=size_to_reach
    )
    symbols = get_optimal_symbols(asp)
    qubit_sets: list[set[Qubit]] = [set() for _ in range(num_partitions)]
    for symbol in symbols:
        if symbol.name != "qubit_in_partition":
            continue
        qubit_idx = symbol.arguments[0].number
        partition_idx = symbol.arguments[1].number
        qubit_sets[partition_idx].add(dag.qubits[qubit_idx])

    decompose_qubit_sets(dag, qubit_sets)


def _qcg_partition_asp(num_partitions: int, size_to_reach: int) -> str:
    asp = f"""
    partition(P) :- P = 0..{num_partitions - 1}.
    
    {{ qubit_in_partition(Qubit, P) : partition(P) }} == 1 :- qubit(Qubit).
    :- partition(P), not qubit_in_partition(_, P).

    virtual_connection(Qubit1, Qubit2, W) :- 
        qubit_in_partition(Qubit1, P1),
        qubit_in_partition(Qubit2, P2),
        Qubit1 != Qubit2,
        P1 != P2,
        qubit_conn(Qubit1, Qubit2, W).

    num_vgates(N) :- N = #sum{{ W, Qubit1, Qubit2 : virtual_connection(Qubit1, Qubit2, W) }}.
    
    num_qubits_in_partition(P, N) :- partition(P), N = #count{{Qubit : qubit_in_partition(Qubit, P)}}.
    :- num_qubits_in_partition(P, N), N > {size_to_reach}.
    
    #minimize{{ N : num_vgates(N) }}.
    #show qubit_in_partition/2.
    """
    return asp
