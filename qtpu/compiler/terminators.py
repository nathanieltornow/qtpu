from typing import Callable
from functools import partial

import numpy as np
import cotengra as ctg

from qtpu.compiler.compress import CompressedIR
from qtpu.compiler.success import success_probability_static
from qtpu.compiler.util import get_leafs


def reach_num_qubits(
    num_qubits: int,
) -> Callable[[CompressedIR, ctg.ContractionTree], bool]:

    def _check_num_qubits(ir: CompressedIR, tree: ctg.ContractionTree) -> bool:
        return all(ir.num_qubits(leaf) <= num_qubits for leaf in get_leafs(tree))

    return _check_num_qubits


def reach_success_prob(
    success_prob: float,
    sucsess_fn: Callable[[CompressedIR, ctg.ContractionTree], float] | None = None,
) -> Callable[[CompressedIR, ctg.ContractionTree], bool]:

    if sucsess_fn is None:
        success_fn = success_probability_static

    def _check_success_prob(ir: CompressedIR, tree: ctg.ContractionTree) -> bool:
        return success_fn(ir, tree) >= success_prob

    return _check_success_prob
