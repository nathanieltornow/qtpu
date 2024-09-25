import cotengra as ctg


def get_leafs(tree: ctg.ContractionTree) -> list[frozenset[int]]:
    nodes = frozenset(range(tree.N))
    for node in tree.childless:
        nodes -= node

    additional = [frozenset([node]) for node in nodes]
    return list(tree.childless) + additional
