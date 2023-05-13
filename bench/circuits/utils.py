import networkx as nx
from qiskit_optimization.applications import Maxcut
from qiskit_optimization import QuadraticProgram


def get_examplary_max_cut_qp(n_nodes: int, degree: int = 2) -> QuadraticProgram:
    """Returns a quadratic problem formulation of a max cut problem of a random graph.

    Keyword arguments:
    n_nodes -- number of graph nodes (and also number of qubits)
    degree -- edges per node
    """
    graph = nx.random_regular_graph(d=degree, n=n_nodes, seed=111)
    maxcut = Maxcut(graph)
    return maxcut.to_quadratic_program()
