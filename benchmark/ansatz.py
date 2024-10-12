import random
import itertools

import networkx as nx

import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import EfficientSU2, RealAmplitudes, ZZFeatureMap, TwoLocal
from qiskit.circuit.random import random_circuit


def vqe(num_qubits: int, reps: int = 2) -> QuantumCircuit:
    circuit = EfficientSU2(num_qubits, entanglement="linear", reps=reps)
    return circuit.assign_parameters(
        {param: 0.4 for param in circuit.parameters}
    ).decompose(), {"name": "vqe", "num_qubits": num_qubits, "reps": reps}


def qml(num_qubits: int, reps: int = 2) -> QuantumCircuit:
    circuit = TwoLocal(
        num_qubits, ["ry", "rx"], "rzz", reps=reps, entanglement="pairwise"
    )
    # circuit = ZZFeatureMap(num_qubits, reps=reps, entanglement="pairwise")
    return circuit.assign_parameters(
        {param: 0.4 for param in circuit.parameters}
    ).decompose(), {"name": "qml", "num_qubits": num_qubits, "reps": reps}


def random_ansatz(num_qubits: int, reps: int) -> QuantumCircuit:
    return random_circuit(num_qubits, reps, max_operands=2), {
        "name": "random",
        "num_qubits": num_qubits,
        "reps": reps,
    }


def qaoa(edges: list[tuple[int, int]], depth: int) -> QuantumCircuit:

    graph = nx.Graph()
    graph.add_edges_from(edges)

    circuit = QuantumCircuit(graph.number_of_nodes())
    for d in range(depth):
        for i in graph.nodes:
            circuit.rx(0.4, i)
            circuit.rz(0.4, i)

        cur_edges = edges if d % 2 == 0 else reversed(edges)

        for i, j in cur_edges:
            circuit.rzz(np.random.rand() * 2 * np.pi, i, j)
    return circuit


def qaoa1(r: int, n: int, m: int, reps: int = 1):
    edges = generate_seperator_graph(r, n, m)
    circ = qaoa(edges, reps)
    return circ, {
        "name": "qaoa1",
        "num_qubits": circ.num_qubits,
        "r": r,
        "n": n,
        "m": m,
        "reps": reps,
    }


def qaoa2(r: int, n: int, m: int, reps: int = 1):
    edges = generate_clustered_graph(r, n, m)
    circuit = qaoa(edges, reps)
    return circuit, {
        "name": "qaoa2",
        "num_qubits": circuit.num_qubits,
        "r": r,
        "n": n,
        "m": m,
        "reps": reps,
    }


def _cluster(
    num_qubits: int, prob: float = 0.8, seed: int | None = None
) -> QuantumCircuit:

    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    rng = np.random.default_rng(seed)

    cluster = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        cluster.rz(rng.random() * np.pi / 2, i)
        cluster.ry(rng.random() * np.pi / 2, i)
    for q1, q2 in itertools.combinations(range(num_qubits), 2):
        if rng.random() < prob:
            cluster.cx(q1, q2)

    circuit = QuantumCircuit(num_qubits)
    circuit.append(cluster.to_gate(), range(num_qubits))
    return circuit


def generate_seperator_graph(
    r: int,
    n: int,
    m: int,
    p_intra: float = 0.7,
    p_inter: float = 0.3,
) -> list[tuple[int, int]]:

    # Initialize an empty graph
    G = nx.Graph()

    # Create clusters and add them to the graph
    cluster_edges: list[list[tuple[int, int]]] = []
    clusters = []
    for i in range(r):
        # Create a random graph for each cluster
        cluster = nx.erdos_renyi_graph(n, p_intra)

        # Relabel nodes to avoid overlap
        mapping = {node: node + i * n for node in cluster.nodes()}
        cluster = nx.relabel_nodes(cluster, mapping)

        # Add the cluster to the main graph
        G = nx.compose(G, cluster)

        # Store the nodes in each cluster
        clusters.append(list(cluster.nodes()))
        cluster_edges.append(list(cluster.edges()))

    # Add multiple vertex separators between each cluster
    separators = []
    for i in range(r - 1):
        sep_nodes = []
        for j in range(m):
            separator = max(G.nodes()) + 1
            sep_nodes.append(separator)
            G.add_node(separator)

            # Connect the separator to nodes in the two clusters it separates
            for node in clusters[i]:
                if np.random.rand() < p_inter:
                    G.add_edge(separator, node)
                    cluster_edges[i].append((separator, node))

            for node in clusters[i + 1]:
                if np.random.rand() < p_inter:
                    G.add_edge(separator, node)
                    cluster_edges[i + 1].append((separator, node))

        separators.append(sep_nodes)

    print(separators)

    return list(itertools.chain.from_iterable(cluster_edges))


def generate_clustered_graph(
    r: int, n: int, m: int, p_intra: int = 0.7
) -> list[tuple[int, int]]:

    # Initialize an empty graph
    G = nx.Graph()

    cluster_edges = []
    # Create clusters and add them to the graph
    clusters = []
    for i in range(r):
        # Create a random graph for each cluster
        cluster = nx.erdos_renyi_graph(n, p_intra)

        # Relabel nodes to avoid overlap
        mapping = {node: node + i * n for node in cluster.nodes()}
        cluster = nx.relabel_nodes(cluster, mapping)

        # Add the cluster to the main graph
        G = nx.compose(G, cluster)

        # Store the nodes in each cluster
        clusters.append(list(cluster.nodes()))
        cluster_edges.append(list(cluster.edges()))

    # Add exactly m connections between each cluster
    for i in range(r - 1):
        connections = 0
        while connections < m:
            # Randomly choose a node from each of the two clusters
            node_from_cluster_1 = random.choice(clusters[i])
            node_from_cluster_2 = random.choice(clusters[i + 1])

            # If there's no edge between these nodes, add one
            if not G.has_edge(node_from_cluster_1, node_from_cluster_2):
                G.add_edge(node_from_cluster_1, node_from_cluster_2)
                cluster_edges[i].append((node_from_cluster_1, node_from_cluster_2))
                connections += 1

    return list(itertools.chain.from_iterable(cluster_edges))
