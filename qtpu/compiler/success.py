from typing import Callable

import cotengra as ctg
from qtpu.compiler.compress import CompressedIR
from qtpu.compiler.util import get_leafs


def success_probability_static(ir: CompressedIR, tree: ctg.ContractionTree) -> float:
    return min((1 - 1e-4) ** len(ir.decompress_nodes(leaf)) for leaf in get_leafs(tree))


def success_num_nodes(ir: CompressedIR, tree: ctg.ContractionTree) -> int:
    return -max(len(ir.decompress_nodes(leaf)) for leaf in get_leafs(tree))


def success_num_qubits(ir: CompressedIR, tree: ctg.ContractionTree) -> int:
    return -max(len(ir.num_qubits(leaf)) for leaf in get_leafs(tree))


def success_reach_qubits(
    max_qubits: int,
) -> Callable[[CompressedIR, ctg.ContractionTree], float]:
    def _func(ir: CompressedIR, tree: ctg.ContractionTree) -> float:
        return (
            0.0
            if max(ir.num_qubits(leaf) for leaf in get_leafs(tree)) > max_qubits
            else 1.0
        )

    return _func
