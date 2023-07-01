import networkx as nx
from networkx.algorithms.community import kernighan_lin_bisection

from qiskit.circuit import Qubit

from _asp import get_optimal_symbols, qcg_to_asp


def bisect_qcg_optimal(qcg: nx.Graph) -> tuple[set[int], set[int]]:
    asp = qcg_to_asp(qcg)
    asp += _bisection_asp()
    symbols = get_optimal_symbols(asp)
    qubit_sets: tuple[set[int], set[int]] = (set(), set())
    for symbol in symbols:
        if symbol.name != "qubit_in_partition":
            continue
        node_idx = symbol.arguments[0].number
        partition_idx = symbol.arguments[1].number
        qubit_sets[partition_idx].add(node_idx)
    return qubit_sets


def bisect_qcg_kl(qcg: nx.Graph) -> tuple[set[Qubit], set[Qubit]]:
    return kernighan_lin_bisection(qcg)


def _bisection_asp() -> str:
    asp = f"""
    partition(0).
    partition(1).
    
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
    
    #minimize{{ |N0 - N1| : num_qubits_in_partition(0, N0), num_qubits_in_partition(1, N1) }}.
    
    #minimize{{ 100@N : num_vgates(N) }}.
    #show qubit_in_partition/2.
    """
    return asp
