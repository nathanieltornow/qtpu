import abc

import cotengra as ctg

from qvm.compiler.compress import CompressedIR
from qvm.compiler.util import get_leafs


class LeafOracle:
    def choose_leaf(
        self, ir: CompressedIR, tree: ctg.ContractionTree
    ) -> frozenset[int] | None:
        return next(get_leafs(tree))


class NumQubitsOracle(LeafOracle):
    def __init__(self, num_qubits: int) -> None:
        self._min_qubits = num_qubits

    def choose_leaf(
        self, ir: CompressedIR, tree: ctg.ContractionTree
    ) -> frozenset[int] | None:
        largest_leaf = max(get_leafs(tree), key=lambda leaf: ir.num_qubits(set(leaf)))
        if ir.num_qubits(set(largest_leaf)) > self._min_qubits:

            return largest_leaf
        return None
