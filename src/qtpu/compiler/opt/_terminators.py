from __future__ import annotations

from typing import TYPE_CHECKING

from qtpu.compiler._util import get_leafs

if TYPE_CHECKING:
    from collections.abc import Callable

    import cotengra as ctg

    from qtpu.compiler._compress import CompressedIR


def reach_num_qubits(
    num_qubits: int,
) -> Callable[[CompressedIR, ctg.ContractionTree], bool]:

    def _check_num_qubits(ir: CompressedIR, tree: ctg.ContractionTree) -> bool:
        return all(ir.num_qubits(set(leaf)) <= num_qubits for leaf in get_leafs(tree))

    return _check_num_qubits
