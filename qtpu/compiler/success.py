import cotengra as ctg
from qtpu.compiler.compress import CompressedIR
from qtpu.compiler.util import get_leafs


def success_probability_static(ir: CompressedIR, tree: ctg.ContractionTree) -> float:
    return min((1 - 1e-4) ** len(ir.decompress_nodes(leaf)) for leaf in get_leafs(tree))


def success_num_nodes(ir: CompressedIR, tree: ctg.ContractionTree) -> int:
    return -max(len(ir.decompress_nodes(leaf)) for leaf in get_leafs(tree))


def success_num_qubits(ir: CompressedIR, tree: ctg.ContractionTree) -> int:
    return -max(len(ir.num_qubits(leaf)) for leaf in get_leafs(tree))
