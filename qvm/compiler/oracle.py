import abc

import cotengra as ctg

from qvm.ir import HybridCircuitIface
from qvm.compiler.util import get_leafs


class Oracle:
    def choose_leaf(
        self, ir: HybridCircuitIface, tree: ctg.ContractionTree
    ) -> frozenset[int] | None:
        return next(get_leafs(tree))


class MaxQubitsOracle(Oracle):
    def __init__(self, min_qubits: int) -> None:
        self._min_qubits = min_qubits

    def choose_leaf(
        self, ir: HybridCircuitIface, tree: ctg.ContractionTree
    ) -> frozenset[int] | None:
        for leaf in get_leafs(tree):
            if ir.num_qubits(set(leaf)) > self._min_qubits:
                return leaf
        return None
