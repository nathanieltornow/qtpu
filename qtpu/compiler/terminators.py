from typing import Callable

import cotengra as ctg

from qtpu.compiler.compress import CompressedIR
from qtpu.compiler.util import get_leafs


def reach_num_qubits(
    num_qubits: int,
) -> Callable[[CompressedIR, ctg.ContractionTree], bool]:

    def _check_num_qubits(ir: CompressedIR, tree: ctg.ContractionTree) -> bool:
        return all(ir.num_qubits(leaf) <= num_qubits for leaf in get_leafs(tree))

    return _check_num_qubits
