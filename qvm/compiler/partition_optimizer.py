import abc
import itertools
from typing import Callable

import cotengra as ctg

from qvm.ir import HybridCircuitIface
from qvm.tensor import HybridTensorNetwork

from qvm.compiler.oracle import Oracle


class Optimizer:
    """
    This class is largely taken and modified from cotengra.core. Thanks to the cotengra contributors for the implementation.
    """

    def __init__(self, oracle: Oracle, max_cost: int):
        self.partition_fn = (
            ctg.pathfinders.path_kahypar.kahypar_subgraph_find_membership
        )
        self.oracle = oracle
        self.max_cost = max_cost

    def optimize(
        self,
        ir: HybridCircuitIface,
        random_strength=0.01,
        # cutoff=10,
        parts=2,
        parts_decay=0.5,
        super_optimize="auto-hq",
        check=False,
        seed=None,
        **partition_opts,
    ) -> ctg.ContractionTree:

        inputs, output, size_dict = ir.inputs(), ir.output(), ir.size_dict()

        tree = ctg.ContractionTree(
            inputs, output, size_dict, track_childless=True, track_flops=True
        )

        rng = ctg.core.get_rng(seed)
        rand_size_dict = ctg.core.jitter_dict(size_dict, random_strength, rng)

        dynamic_imbalance = ("imbalance" in partition_opts) and (
            "imbalance_decay" in partition_opts
        )
        if dynamic_imbalance:
            imbalance = partition_opts.pop("imbalance")
            imbalance_decay = partition_opts.pop("imbalance_decay")
        else:
            imbalance = imbalance_decay = None

        dynamic_fix = partition_opts.get("fix_output_nodes", None) == "auto"

        while tree_node := self.oracle.choose_leaf(ir=ir, tree=tree):
            subgraph = tuple(tree_node)
            subsize = len(subgraph)

            # relative subgraph size
            s = subsize / tree.N

            # let the target number of communities depend on subgraph size
            parts_s = max(int(s**parts_decay * parts), 2)

            # let the imbalance either rise or fall
            if dynamic_imbalance:
                if imbalance_decay >= 0:
                    imbalance_s = s**imbalance_decay * imbalance
                else:
                    imbalance_s = 1 - s**-imbalance_decay * (1 - imbalance)
                partition_opts["imbalance"] = imbalance_s

            if dynamic_fix:
                # for the top level subtree (s==1.0) we partition the outputs
                # nodes first into their own bi-partition
                parts_s = 2
                partition_opts["fix_output_nodes"] = s == 1.0

            # partition! get community membership list e.g.
            # [0, 0, 1, 0, 1, 0, 0, 2, 2, ...]
            inputs = tuple(map(tuple, tree.node_to_terms(subgraph)))
            output = tuple(tree.get_legs(tree_node))
            membership = self.partition_fn(
                inputs,
                output,
                rand_size_dict,
                parts=parts_s,
                seed=rng,
                **partition_opts,
            )

            # divide subgraph up e.g. if we enumerate the subgraph index sets
            # (0, 1, 2, 3, 4, 5, 6, 7, 8, ...) ->
            # ({0, 1, 3, 5, 6}, {2, 4}, {7, 8})
            new_subgs = tuple(
                map(ctg.core.node_from_seq, ctg.core.separate(subgraph, membership))
            )

            if len(new_subgs) == 1:
                continue

            # update tree structure with newly contracted subgraphs
            tree.contract_nodes(new_subgs, optimize=super_optimize, check=check)


            if tree.contraction_cost() > self.max_cost:
                tree.children.pop(tree_node)
                break

        return tree


# import networkx as nx
# import cotengra as ctg
# from cotengra.pathfinders.path_kahypar import kahypar_subgraph_find_membership
# from cotengra.pathfinders.path_igraph import igraph_subgraph_find_membership
# from cotengra.pathfinders.path_labels import labels_partition


# def partition_kahypar(
#     inputs: list[tuple[str, ...]],
#     output: tuple[str, ...],
#     size_dict: dict[str, int],
#     parts: int,
# ) -> list[int]:
#     return kahypar_subgraph_find_membership(inputs, output, size_dict, parts)


# def partition_igraph(
#     inputs: list[tuple[str, ...]],
#     output: tuple[str, ...],
#     size_dict: dict[str, int],
#     parts: int,
# ) -> list[int]:
#     return igraph_subgraph_find_membership(inputs, output, size_dict, parts)


# def partition_labels(
#     inputs: list[tuple[str, ...]],
#     output: tuple[str, ...],
#     size_dict: dict[str, int],
#     parts: int,
# ) -> list[int]:
#     return labels_partition(inputs, output, size_dict, parts)


# def partition_girvan_newman(
#     inputs: list[tuple[str, ...]],
#     output: tuple[str, ...],
#     size_dict: dict[str, int],
#     parts: int,
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
#             membership[node] = i
#     return membership


# def membership_to_subsets(membership: list[int]) -> list[set[int]]:
#     subsets = {}
#     for i, group in enumerate(membership):
#         subsets.setdefault(group, set()).add(i)
#     return list(subsets.values())


# def partition_kernighan_lin(
#     inputs: list[tuple[str, ...]],
#     output: tuple[str, ...],
#     size_dict: dict[str, int],
#     parts: int,
# ) -> list[int]:
#     hypergraph = ctg.HyperGraph(inputs, output, size_dict)
#     graph = hypergraph.to_networkx()
#     for _, _, data in graph.edges(data=True):
#         data["weight"] = size_dict[data["ind"]]

#     A, B = nx.algorithms.community.kernighan_lin_bisection(graph)

#     return membership
# )


# class Optimizer(abc.ABC):
#     @abc.abstractmethod
#     def optimize(self, ir: HybridCircuitIface) -> HybridTensorNetwork:
#         """
#         Optimizes a hybrid circuit, and returns the optimized hybrid tensor network.

#         Args:
#             ir (HybridCircuitIface): An interface to the hybrid circuit IR.

#         Returns:
#             HybridTensorNetwork: The optimized hybrid tensor network.
#         """
#         ...


# class GreedyOptimizer(Optimizer, abc.ABC):

#     def __init__(self, bisection_method: str = "gn") -> None:
#         super().__init__()
#         match bisection_method:
#             case "gn":
#                 self._bisect_func = partition_girvan_newman
#             case _:
#                 raise ValueError(f"Unknown bisection method: {bisection_method}")

#     def bisect(
#         self, inputs: list[tuple[str, ...]], size_dict: dict[str, int]
#     ) -> tuple[set[int], set[int]]:
#         membership = self._bisect_func(inputs, (), size_dict, 2)
#         A, B = membership_to_subsets(membership)
#         return A, B

#     @abc.abstractmethod
#     def next_leaf(
#         self, ir: HybridCircuitIface, current_tree: ctg.ContractionTree
#     ) -> frozenset[int] | None: ...

#     def optimize(self, ir: HybridCircuitIface) -> HybridTensorNetwork:
#         tree = ctg.ContractionTree(ir.inputs(), ir.output(), ir.size_dict())
#         while (leaf := self.next_leaf(ir, tree)) is not None:
#             inputs = []
#             A, B = self.bisect(ir.inputs(), ir.size_dict())
#             tree.contract_nodes_pair(frozenset(A), frozenset(B))
#         return ir.hybrid_tn(tree.node_subsets())
