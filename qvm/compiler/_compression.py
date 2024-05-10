import abc
from dataclasses import dataclass
import networkx as nx
from qiskit.circuit import Qubit

from qvm.graph import CircuitGraphNode, CircuitGraph


@dataclass
class CompressedNode:
    op_id: int
    nodes: set[CircuitGraphNode]

    @property
    def qubits(self) -> set[Qubit]:
        return {node.qubit for node in self.nodes}

    def __hash__(self) -> int:
        return hash(self.op_id)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, CompressedNode):
            return False
        return self.op_id == value.op_id


def decompress_nodes(nodes: set[CompressedNode]) -> set[CircuitGraphNode]:
    return {node for compressed_node in nodes for node in compressed_node.nodes}


def compress_nothing(cg: CircuitGraph) -> nx.Graph:
    nodes = {
        hash((node.op_id, node.qubit)): CompressedNode(
            hash((node.op_id, node.qubit)), {node}
        )
        for node in cg.graph.nodes
    }
    edges = [
        (
            nodes[hash((node1.op_id, node1.qubit))],
            nodes[hash((node2.op_id, node2.qubit))],
            weight,
        )
        for node1, node2, weight in cg.graph.edges(data="weight")
    ]

    graph = nx.Graph()
    graph.add_nodes_from(nodes.values())
    graph.add_weighted_edges_from(edges)
    return graph


def compress_qubits(cg: CircuitGraph) -> nx.Graph:
    compressed_nodes = {
        qubit: CompressedNode(i, set()) for i, qubit in enumerate(cg.circuit.qubits)
    }
    compressed_graph = nx.Graph()
    compressed_graph.add_nodes_from(compressed_nodes.values())

    uncompressed = cg.graph

    for node in uncompressed.nodes:
        compressed_nodes[node.qubit].nodes.add(node)

    for node1, node2, weight in uncompressed.edges(data="weight"):
        if node1.qubit == node2.qubit:
            continue

        compr_node1, compr_node2 = (
            compressed_nodes[node1.qubit],
            compressed_nodes[node2.qubit],
        )

        if not compressed_graph.has_edge(compr_node1, compr_node2):
            compressed_graph.add_edge(
                compr_node1,
                compr_node2,
                weight=0,
            )
        # update weight
        compressed_graph[compr_node1][compr_node2]["weight"] += weight

    return compressed_graph
