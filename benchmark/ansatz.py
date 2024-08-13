import random
import itertools

import networkx as nx

import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import EfficientSU2, RealAmplitudes, ZZFeatureMap, TwoLocal
from qiskit.circuit.random import random_circuit


def generate_ansatz(name: str, num_qubits: int, reps: int) -> QuantumCircuit:
    if name == "linear":
        return linear_ansatz(num_qubits, reps)
    elif name == "brick":
        return brick_ansatz(num_qubits, reps)
    elif name == "zz":
        return zzfeaturemap_ansatz(num_qubits, reps)
    elif name == "rnd":
        return random_ansatz(num_qubits, reps)
    elif name == "clu":
        return cluster_ansatz(num_qubits, reps)
    else:
        raise ValueError(f"Unknown ansatz name: {name}")


def linear_ansatz(num_qubits: int, reps: int) -> QuantumCircuit:
    circuit = EfficientSU2(num_qubits, entanglement="linear", reps=reps)
    return circuit.assign_parameters(
        {param: np.random.rand() * np.pi / 2 for param in circuit.parameters}
    ).decompose()


def brick_ansatz(num_qubits: int, reps: int) -> QuantumCircuit:
    circuit = TwoLocal(
        num_qubits, ["ry", "rz"], "rzz", entanglement="pairwise", reps=reps
    )
    return circuit.assign_parameters(
        {param: np.random.rand() * np.pi / 2 for param in circuit.parameters}
    ).decompose()


def zzfeaturemap_ansatz(num_qubits: int, reps: int) -> QuantumCircuit:
    circuit = ZZFeatureMap(num_qubits, reps=reps, entanglement="pairwise")
    return circuit.assign_parameters(
        {param: np.random.rand() * np.pi / 2 for param in circuit.parameters}
    ).decompose()


def random_ansatz(num_qubits: int, reps: int) -> QuantumCircuit:
    return random_circuit(num_qubits, reps, max_operands=2)


def cluster_ansatz(num_qubits: int, reps: int, seed: int = None) -> QuantumCircuit:
    # fill clusters with random numbers between 15 - 20 qubits each, and the last cluster with the remaining qubits
    cluster_sizes = np.random.randint(15, 20, num_qubits // 20)
    last_cluster = [num_qubits - sum(cluster_sizes)]
    if last_cluster[0] > 20:
        last_cluster = [20, last_cluster[0] - 20]
    cluster_sizes: list[int] = list(cluster_sizes) + last_cluster

    print(cluster_sizes)

    cluster_regs = [QuantumRegister(s, f"q_{i}") for i, s in enumerate(cluster_sizes)]
    circuit = QuantumCircuit(*cluster_regs)

    for _ in range(reps):
        for qreg in cluster_regs:
            circuit.compose(_cluster(qreg.size, 0.8, seed), qubits=qreg, inplace=True)

        for i in range(len(cluster_sizes) - 1):
            circuit.cx(cluster_regs[i][-1], cluster_regs[i + 1][0])
    return circuit


def qaoa(edges: list[tuple[int, int]], depth: int) -> QuantumCircuit:

    graph = nx.Graph()
    graph.add_edges_from(edges)

    circuit = QuantumCircuit(graph.number_of_nodes())
    for d in range(depth):
        for i in graph.nodes:
            circuit.rx(np.random.rand() * 2 * np.pi, i)
            circuit.rz(np.random.rand() * 2 * np.pi, i)

        cur_edges = edges if d % 2 == 0 else reversed(edges)

        for i, j in cur_edges:
            circuit.rzz(np.random.rand() * 2 * np.pi, i, j)
    return circuit


def qaoa1(r: int, n: int, m: int):
    edges = generate_seperator_graph(r, n, m)
    return qaoa(edges, 1)


def qaoa2(r: int, n: int, m: int):
    edges = generate_clustered_graph(r, n, m)
    return qaoa(edges, 2)


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
