from typing import Iterator
from itertools import permutations

import networkx as nx
from qiskit.circuit import Qubit, CircuitInstruction, Reset

from .dag import DAG


def qubit_reuse(dag: DAG, size_to_reach: int = 1) -> None:
    num_qubits = len(dag.qubits)
    while num_qubits > size_to_reach:
        qubit_pair = next(find_valid_reuse_pairs(dag), None)
        if qubit_pair is None:
            break
        reuse(dag, *qubit_pair)
        dag.compact()
        num_qubits -= 1


def reuse(dag: DAG, qubit: Qubit, reused_qubit: Qubit) -> None:
    """
    Reuse a qubit by resetting it and reusing it.
    NOTE: Only works if qubit is not dependent on reused_qubit.
        This must be checked by the caller.

    Args:
        dag (DAG): The DAG to modify.
        qubit (Qubit): The qubit.
        reused_qubit (Qubit): The qubit to reuse.
    """

    # first op of u_qubit
    first_node = next(dag.nodes_on_qubit(reused_qubit))
    # last op of v_qubit
    last_node = list(dag.nodes_on_qubit(qubit))[-1]

    reset_instr = CircuitInstruction(operation=Reset(), qubits=(reused_qubit,))
    reset_node = dag.add_instr_node(reset_instr)
    dag.add_edge(last_node, reset_node)
    dag.add_edge(reset_node, first_node)

    for node in dag.nodes:
        instr = dag.get_node_instr(node)
        instr.qubits = [
            reused_qubit if instr_qubit == qubit else instr_qubit
            for instr_qubit in instr.qubits
        ]


def is_dependent_qubit(dag: DAG, u_qubit: Qubit, v_qubit: Qubit) -> bool:
    """Checks if any operation on u_qubit depends on any operation on v_qubit.

    Args:
        dag (DAG): The DAG to check.
        u_qubit (Qubit): The first qubit.
        v_qubit (Qubit): The second qubit.

    Returns:
        bool: Whether any operation on u_qubit depends on any operation on v_qubit.
    """
    # first op of u_qubit
    u_node = next(dag.nodes_on_qubit(u_qubit))
    # last op of v_qubit
    v_node = list(dag.nodes_on_qubit(v_qubit))[-1]
    return nx.has_path(dag, u_node, v_node)


def find_valid_reuse_pairs(dag: DAG) -> Iterator[tuple[Qubit, Qubit]]:
    """Finds all valid reuse pairs in a DAG by trying every possible pair. O(n^2).

    Args:
        dag (DAG): The DAG to check.

    Yields:
        Iterator[tuple[Qubit, Qubit]]: All valid reuse pairs.
    """
    for qubit, reused_qubit in permutations(dag.qubits, 2):
        if not is_dependent_qubit(dag, reused_qubit, qubit):
            yield qubit, reused_qubit
