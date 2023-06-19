from typing import Iterator
from itertools import combinations

import networkx as nx
from qiskit.circuit import Qubit, CircuitInstruction, Reset

from qvm.dag import DAG


def reuse(dag: DAG, qubit: Qubit, reused_qubit: Qubit) -> None:
    """
    Reuse a qubit by resetting it and reusing it.
    IMPORTANT: Only works if qubit is not dependent on reused_qubit.
        This must be checked by the caller.

    Args:
        dag (DAG): The DAG to modify.
        qubit (Qubit): The qubit.
        reused_qubit (Qubit): The qubit to reuse.
    """

    # first op of u_qubit
    first_node = next(dag.instructions_on_qubit(reused_qubit))
    # last op of v_qubit
    last_node = list(dag.instructions_on_qubit(qubit))[-1]

    reset_instr = CircuitInstruction(operation=Reset(), qubits=(first_node,))
    reset_node = dag.add_instr_node(reset_instr)
    dag.add_edge(last_node, reset_node)
    dag.add_edge(reset_node, first_node)

    for node in dag.nodes:
        instr = dag.get_node_instr(node)
        instr.qubits = [
            reused_qubit if qubit == qubit else qubit for qubit in instr.qubits
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
    u_node = next(dag.instructions_on_qubit(u_qubit))
    # last op of v_qubit
    v_node = list(dag.instructions_on_qubit(v_qubit))[-1]
    return nx.has_path(dag, u_node, v_node)


def find_valid_reuse_pairs(dag: DAG) -> Iterator[tuple[Qubit, Qubit]]:
    """Finds all valid reuse pairs in a DAG by trying every possible pair. O(n^2).

    Args:
        dag (DAG): The DAG to check.

    Yields:
        Iterator[tuple[Qubit, Qubit]]: All valid reuse pairs.
    """
    for from_qubit, to_qubit in combinations(dag.qubits, 2):
        if not is_dependent_qubit(dag, from_qubit, to_qubit):
            yield from_qubit, to_qubit
