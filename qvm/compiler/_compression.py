import abc
from dataclasses import dataclass
import networkx as nx
from qiskit.circuit import Qubit, QuantumCircuit

from qvm.graph import CircuitGraphNode, CircuitGraph


@dataclass
class CompressedNode:
    node_id: int
    nodes: set[CircuitGraphNode]

    @property
    def qubits(self) -> set[Qubit]:
        return {node.qubit for node in self.nodes}

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, CompressedNode):
            return False
        return self.node_id == value.node_id


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


def remove_1q_gates(cg: CircuitGraph) -> nx.Graph:
    compr_graph = nx.Graph()

    qubit_to_nodes = {}
    initial_nodes = {qubit: set() for qubit in cg.circuit.qubits}

    for op_id, instr in enumerate(cg.circuit):
        qubits = instr.qubits
        if len(qubits) == 1:
            qubit = qubits[0]
            if qubit in initial_nodes:
                initial_nodes[qubit].add(CircuitGraphNode(op_id, qubit))
                continue

            qubit_to_nodes[qubit].nodes.add(CircuitGraphNode(op_id, qubit))
            continue

        assert len(qubits) == 2

        new_nodes = [
            CompressedNode(hash((op_id, qubit)), {CircuitGraphNode(op_id, qubit)})
            for qubit in qubits
        ]
        compr_graph.add_nodes_from(new_nodes)
        compr_graph.add_edge(new_nodes[0], new_nodes[1], weight=5)

        for qubit, new_node in zip(qubits, new_nodes):
            if qubit in initial_nodes:
                nodes = initial_nodes.pop(qubit)
                new_node.nodes.update(nodes)
                qubit_to_nodes[qubit] = new_node
                continue

            compr_graph.add_edge(qubit_to_nodes[qubit], new_node, weight=4)
            qubit_to_nodes[qubit] = new_node

    return compr_graph


# def compress(cg: CircuitGraph) -> nx.Graph:
#     runs: dict[Qubit, set[CircuitGraphNode]] = {
#         qubit: set() for qubit in cg.circuit.qubits
#     }
#     qubit_conns: dict[Qubit, Qubit] = {}

#     for instr in cg.circuit:
#         qubits = instr.qubits

#         if len(qubits) == 1:
#             runs[qubits[0]].add(CircuitGraphNode(instr.op_id, qubits[0]))
#             continue

#         assert len(qubits) == 2

#         if qubits[0] not in qubit_conns:
#             qubit_conns[qubits[0]] = qubits[1]
#             qubit_conns[qubits[1]] = qubits[0]
#             runs[qubits[0]].add(CircuitGraphNode(instr.op_id, qubits[0]))
#             runs[qubits[1]].add(CircuitGraphNode(instr.op_id, qubits[1]))

#         elif qubit_conns[qubits[0]] == qubits[1]:
#             runs[qubits[0]].add(CircuitGraphNode(instr.op_id, qubits[0]))
#             runs[qubits[1]].add(CircuitGraphNode(instr.op_id, qubits[1]))
#             continue



# def compress_hybrid(cg: CircuitGraph) -> nx.DiGraph:
#     current_nodes: dict[Qubit, CompressedNode] = {}
#     qubit_interactions: dict[Qubit, Qubit] = {
#         qubit: qubit for qubit in cg.circuit.qubits
#     }

#     compr = nx.Graph()

#     def _append_to_current_node(node: CircuitGraphNode) -> CompressedNode:
#         if node.qubit not in current_nodes:
#             new_node = CompressedNode(hash((node.op_id, node.qubit)), set())
#             current_nodes[node.qubit] = new_node
#             compr.add_node(new_node)
#         current_nodes[node.qubit].nodes.add(node)
#         return current_nodes[node.qubit]

#     def _add_new_node(node: CircuitGraphNode) -> CompressedNode:
#         if node.qubit not in current_nodes:
#             return _append_to_current_node(node)

#         new_node = CompressedNode(hash((node.op_id, node.qubit)), {node})
#         compr.add_node(new_node)
#         compr.add_edge(current_nodes[node.qubit], new_node, weight=4)
#         current_nodes[node.qubit] = new_node
#         return new_node

#     uncompr = cg.graph

#     for node in nx.topological_sort(uncompr):
#         next_node = cg.next_node_operation(node)
#         # prev_node = cg.prev_node_operation(node)

#         if next_node is None:
#             # single qubit gate
#             _append_to_current_node(node)
#             continue

#         other_node = next_node

#         # if qubit_interactions
#         # if qubit_interactions[node.qubit] == other_node.qubit:
#         #     _append_to_current_node(node)
#         #     _append_to_current_node(other_node)
#         #     compr[current_nodes[node.qubit]][current_nodes[other_node.qubit]][
#         #         "weight"
#         #     ] += 5
#         #     continue

#         new_node1 = _add_new_node(node)
#         new_node2 = _add_new_node(other_node)
#         print(new_node1, new_node2)
#         compr.add_edge(new_node1, new_node2, weight=5)

#         qubit_interactions[node.qubit] = other_node.qubit
#         qubit_interactions[other_node.qubit] = node.qubit

#     return compr


# def compress_hybrid(cg: CircuitGraph) -> nx.Graph:

#     current_nodes: dict[Qubit, CompressedNode] = {}

#     qubit_pairs: dict[Qubit, set[Qubit]] = {}

#     compr = nx.Graph()


#     def _add_node(op_id: int, qubit: Qubit):
#         if qubit not in current_nodes:
#             new_node = CompressedNode(hash((op_id, qubit)), set())
#             current_nodes[qubit] = new_node
#             compr.add_node(new_node)
#         current_nodes[qubit].nodes.add(CircuitGraphNode(op_id, qubit))

#     for op_id, instr in enumerate(cg.circuit):
#         qubits = instr.qubits
#         if len(qubits) == 1:
#             _add_node(op_id, qubits[0])
#             continue

#         assert len(qubits) == 2

#         if all(set(qubits) == qubit_pairs.get(qubit, set()) for qubit in qubits):
#             for qubit in qubits:
#                 _add_node(op_id, qubit)

#             node1, node2 = current_nodes[qubits[0]], current_nodes[qubits[1]]
#             if not compr.has_edge(node1, node2):
#                 compr.add_edge(node1, node2, weight=0)
#             compr[node1][node2]["weight"] += 5
#             continue


#         for qubit in qubits:
#             if qubit not in qubit_pairs:
#                 qubit_pairs[qubit] = set(qubits)


#         for qubit in qubits:
#             qubit_pairs[qubit] = set(qubits)


# if set(qubits) == qubit_pairs[qubit]:
#     _add_node(op_id, qubit)
#     continue

# # create new compressed node
# new_node = CompressedNode(op_id, set())


# def compress_hybrid(cg: CircuitGraph) -> nx.DiGraph:
#     compressed = remove_1q_gates(cg)

#     qubit_pairs = set()

#     for instr in cg.circuit:
#         qubits = frozenset(instr.qubits)
#         if len(qubits) == 1:
#             continue

#         assert len(qubits) == 2

#         if qubits in qubit_pairs:


# def compress_hybrid(cg: CircuitGraph) -> nx.Graph:
#     qubit_pairs = {qubit: {qubit, qubit} for qubit in cg.circuit.qubits}

#     compressed_graph = nx.Graph()

#     compr_nodes = {
#         qubit: CompressedNode(i, set()) for i, qubit in enumerate(cg.circuit.qubits)
#     }

#     uncompr_graph = cg.graph

#     for node in nx.topological_sort(cg.graph):
#         next_node = cg.next_node_operation(node)
#         if next_node is None:
#             compr_nodes[node.qubit].nodes.add(node)
#             continue

#         cur_qubit_pair = qubit_pairs[node.qubit]
#         act_qubit_pair = {node.qubit, next_node.qubit}

#         if cur_qubit_pair == act_qubit_pair:
#             compr_nodes[node.qubit].nodes.add(node)

#             compr1, compr2 = compr_nodes[node.qubit], compr_nodes[next_node.qubit]
#             compressed_graph[compr1][compr2]["weight"] += uncompr_graph[node][
#                 next_node
#             ]["weight"]
#             continue

#         compressed_graph.add_node(compr_nodes[node.qubit])
