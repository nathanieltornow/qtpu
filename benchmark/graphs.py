import networkx as nx


def k3():
    return nx.complete_graph(3)


def k4():
    return nx.complete_graph(4)


def k5():
    return nx.complete_graph(5)


def k6():
    return nx.complete_graph(6)


def k7():
    return nx.complete_graph(7)


def barbell(num_nodes: int):
    return nx.barbell_graph(num_nodes, 0)


def erdos_renyi(num_nodes: int, p: float = 0.5):
    return nx.erdos_renyi_graph(num_nodes, p)
