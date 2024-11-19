from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qiskit_addon_cutting.qpd import TwoQubitQPDGate

from qtpu.transforms import wire_cuts_to_moves

if TYPE_CHECKING:
    import cotengra as ctg
    from qiskit.circuit import QuantumCircuit


def get_leafs(tree: ctg.ContractionTree) -> list[frozenset[int]]:
    nodes = frozenset(range(tree.N))
    for node in tree.childless:
        nodes -= node

    additional = [frozenset([node]) for node in nodes]
    return list(tree.childless) + additional


def sampling_overhead_tree(tree: ctg.ContractionTree) -> float:
    involved_inds = set()
    for node in get_leafs(tree):
        involved_inds |= set(tree.get_legs(node))
        # print(tree.get_legs(node))

    # return np.prod([tree.size_dict[ind] for ind in involved_inds])
    return float(sum(np.log10(tree.size_dict[ind]) for ind in involved_inds))


def sampling_overhead_circuit(circuit: QuantumCircuit) -> float:
    circuit = wire_cuts_to_moves(circuit)
    return float(np.prod(
        [
            instr.operation.basis.overhead
            for instr in circuit
            if isinstance(instr.operation, TwoQubitQPDGate)
        ]
    ))


# def partition_girvan_newman(
#     inputs: list[tuple[str, ...]],
#     output: tuple[str, ...],
#     size_dict: dict[str, int],
#     parts: int,
#     **kwargs,
# ) -> list[int]:
#     hypergraph = ctg.HyperGraph(inputs, output, size_dict)

#     graph = hypergraph.to_networkx()
#     for _, _, data in graph.edges(data=True):
#         data["weight"] = size_dict[data["ind"]]

#     for components in nx.algorithms.community.girvan_newman(graph):
#         if len(components) == parts:
#             break

#     membership = [0] * len(inputs)
#     for i, component in enumerate(components):
#         for node in component:
#             if isinstance(node, str):
#                 continue
#             membership[node] = i
#     return membership
