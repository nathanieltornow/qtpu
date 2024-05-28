from itertools import chain

from qiskit.circuit import QuantumCircuit

from qvm.ir import HybridCircuitIR, HybridCircuitIface
from qvm.tensor import HybridTensorNetwork


class CompressedIR(HybridCircuitIface):
    def __init__(self, ir: HybridCircuitIR) -> None:
        self._ir = ir
        self._compressed_nodes = {node: {node} for node in self.hypergraph.nodes}
        self._inputs = ir.inputs.copy()

    def inputs(self) -> list[tuple[str, ...]]:
        return self._inputs

    def output(self) -> tuple[str, ...]:
        return tuple()

    def size_dict(self) -> dict[str, int]:
        return self._ir.hypergraph.size_dict

    def decompress_nodes(self, compressed_nodes: set[int]) -> set[int]:
        return set(
            chain.from_iterable(
                self._compressed_nodes[node] for node in compressed_nodes
            )
        )

    def quantum_tensor(self, compressed_nodes: set[int]) -> QuantumCircuit:
        return self._ir.quantum_tensor(self.decompress_nodes(compressed_nodes))

    def num_qubits(self, compressed_nodes: set[int]) -> int:
        return len(
            set(
                self._ir.node_infos[node].abs_qubit
                for node in self.decompress_nodes(compressed_nodes)
            )
        )

    def compress(self, node1: int, node2: int) -> int:
        node1, node2 = sorted([node1, node2])
        self._compressed_nodes[node1] = (
            self._compressed_nodes[node1] | self._compressed_nodes[node2]
        )
        self._inputs[node1] = tuple(set(self._inputs[node1] + self._inputs[node2]))
        self._compressed_nodes.pop(node2)
        return node1

    def hybrid_tn(self, node_subsets: list[set[int]]) -> HybridTensorNetwork:
        decompressed_subsets = [
            self.decompress_nodes(subset) for subset in node_subsets
        ]
        return self._ir.hybrid_tn(decompressed_subsets)


def compress_qubits(ir: HybridCircuitIR) -> CompressedIR:
    compr = CompressedIR(ir)
    node_per_qubit = {}
    for nodes in ir.op_nodes:
        for node in nodes:
            qubit = ir._node_infos[node].abs_qubit
            if qubit not in node_per_qubit:
                node_per_qubit[qubit] = node
            else:
                compr.compress_nodes(node_per_qubit[qubit], node)
    return compr


# def compress_ops(ir: HybridCircuitIR) -> CompressedIR:
#     compr = CompressedIR(ir)
#     node_per_op = {}
#     for nodes in ir.op_nodes:
#         for node in nodes:
#             op = ir._node_infos[node].op
#             if op not in node_per_op:
#                 node_per_op[op] = node
#             else:
#                 compr.compress_nodes(node_per_op[op], node)
#     return compr


# def compress_1q_gates(ir: HybridCircuitIR) -> CompressedIR:
#     compr = CompressedIR(ir)
#     node_per_qubit = {}
#     for nodes in ir.op_nodes:
#         for node in nodes:
#             qubit = ir._node_infos[node].abs_qubit
#             if qubit not in node_per_qubit:
#                 node_per_qubit[qubit] = node
#             else:
#                 compr.compress_nodes(node_per_qubit[qubit], node)
#     return compr


# def compress_2q_gates(ir: HybridCircuitIR) -> CompressedIR:
#     compr = CompressedIR(ir)
#     node_per_qubit = {}
#     for nodes in ir.op_nodes:
#         qubits = [ir.node_infos[node].abs_qubit for node in nodes]
#         if len(nodes) == 2:
#             compr.compress_nodes(*nodes)
#             node_per_qubit[qubits[0]] = nodes[0]
#             node_per_qubit[qubits[1]] = nodes[1]
