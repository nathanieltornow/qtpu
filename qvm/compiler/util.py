from typing import Iterator

import cotengra as ctg


def traverse(tree: ctg.ContractionTree, order: str = "bfs") -> Iterator[frozenset[int]]:
    queue = [tree.root]
    while queue:
        node = queue.pop(0 if order == "bfs" else -1)
        yield node
        if node in tree.children:
            queue.append(tree.children[node][0])
            queue.append(tree.children[node][1])


def get_leafs(tree: ctg.ContractionTree) -> Iterator[frozenset[int]]:
    for node in traverse(tree):
        if node not in tree.children:
            yield node


def tree_to_hypergraph(tree: ctg.ContractionTree) -> ctg.HyperGraph:
    hypergraph = ctg.HyperGraph(tree.inputs, tree.output, tree.size_dict)
    for leaf_set in get_leafs(tree):
        nodes = list(leaf_set)
        for v in nodes[1:]:
            hypergraph.contract(v, nodes[0])
    return hypergraph
