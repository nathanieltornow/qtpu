from itertools import chain

import cotengra as ctg
from qiskit.circuit import QuantumCircuit

from qtpu.compiler.ir import HybridCircuitIR, NodeInfo


class CompressedIR:
    def __init__(self, ir: HybridCircuitIR, contract_indices: set[str]) -> None:
        self._ir = ir
        hg = ir.hypergraph.copy()

        compressed_nodes: dict[int, tuple[set[int], tuple[str, ...]]] = {
            i: ({i}, hg.inputs[i]) for i in range(hg.num_nodes)
        }

        for idx in contract_indices:

            i, j = hg.edges[idx]
            new_node = hg.contract(i, j)

            compr1, inputs1 = compressed_nodes[i]
            compr2, inputs2 = compressed_nodes[j]

            compr = compr1.union(compr2)

            inputs = (set(inputs1) - set(inputs2)) | (set(inputs2) - set(inputs1))

            new_inputs = tuple(inputs)

            compressed_nodes[new_node] = (compr, new_inputs)
            compressed_nodes.pop(i, None)
            compressed_nodes.pop(j, None)

        self._compressed_nodes = [compr for compr, _ in compressed_nodes.values()]
        self._inputs = [inputs for _, inputs in compressed_nodes.values()]

        self._sizedict = hg.size_dict

    def contraction_tree(self) -> ctg.ContractionTree:
        return ctg.ContractionTree(
            self._inputs,
            tuple(),
            self._sizedict,
            track_childless=True,
            track_flops=True,
        )

    def num_qubits(self, node_subset: set[int]) -> int:
        return len(
            set(
                chain.from_iterable(
                    set(info.abs_qubit for info in self.node_infos(node))
                    for node in node_subset
                )
            )
        )

    def cut_circuit(self, node_subsets: list[set[int]]) -> QuantumCircuit:
        decompressed_nodes = [self.decompress_nodes(subset) for subset in node_subsets]
        return self._ir.cut_circuit(decompressed_nodes)

    def node_infos(self, node: int) -> set[NodeInfo]:
        decompressed_nodes = self.decompress_nodes({node})
        return {self._ir.node_info(node) for node in decompressed_nodes}

    def decompress_nodes(self, compressed_nodes: set[int]) -> set[int]:
        return set(
            chain.from_iterable(
                self._compressed_nodes[node] for node in compressed_nodes
            )
        )

    # def hybrid_tn(self, node_subsets: list[set[int]]) -> HybridTensorNetwork:
    #     decompressed_nodes = [self.decompress_nodes(subset) for subset in node_subsets]
    #     return self._ir.hybrid_tn(decompressed_nodes)

    # def quantum_tensor(self, node_subset: set[int]) -> QuantumTensor:
    #     decompressed_nodes = self.decompress_nodes(node_subset)
    #     return self._ir.quantum_tensor(decompressed_nodes)


def compress_2q_gates(ir: HybridCircuitIR) -> CompressedIR:
    hg = ir.hypergraph
    indices = set()
    for idx, (u, v) in hg.edges.items():
        if (
            all(len(list(hg.neighbors(node))) == 3 for node in [u, v])
            and ir.node_info(u).op_idx == ir.node_info(v).op_idx
        ):
            indices.add(idx)
    return CompressedIR(ir, indices)


def compress_qubits(ir: HybridCircuitIR) -> CompressedIR:
    hg = ir.hypergraph
    indices = set()
    for idx, (u, v) in hg.edges.items():
        if ir.node_info(u).abs_qubit == ir.node_info(v).abs_qubit:
            indices.add(idx)
    return CompressedIR(ir, indices)
