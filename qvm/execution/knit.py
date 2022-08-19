from typing import Dict, List, Optional, Tuple

import lithops.multiprocessing as mp

from qvm.circuit.virtual_gate import VirtualBinaryGate
from qvm.result import Result


class KnitTree:
    virtual_gate: Optional[VirtualBinaryGate]
    result: Optional[Result]
    sub_nodes: Optional[List["KnitTree"]]

    pool: Optional[mp.Pool]

    def __init__(
        self,
        results: Dict[Tuple[int, ...], Result],
        virtual_gates: List[VirtualBinaryGate],
        path: Tuple[int, ...] = (),
        pool: Optional[mp.Pool] = None,
    ) -> None:
        self.pool = pool
        self.result = results.get(path, None)

        # if child node
        if len(virtual_gates) == 0:
            self.sub_nodes = None
            self.virtual_gate = None
            return

        vgate = virtual_gates.pop(0)
        self.virtual_gate = vgate
        self.sub_nodes = [
            KnitTree(results, virtual_gates.copy(), path + (i,))
            for i in range(len(vgate.configure()))
        ]

    def _pool_wrapper(self, sub_node: "KnitTree") -> Result:
        return sub_node.knit()

    def knit(self) -> Result:
        if self.result:
            return self.result
        if self.sub_nodes and self.virtual_gate:
            results: List[Result]
            if self.pool is not None:
                results = self.pool.map(self._pool_wrapper, self.sub_nodes)
            else:
                results = [subnode.knit() for subnode in self.sub_nodes]
            return self.virtual_gate.knit(results)
        raise Exception("cannot knit (unexpected)")


def knit(
    results: Dict[Tuple[int, ...], Result],
    virtual_gates: List[VirtualBinaryGate],
    pool: Optional[mp.Pool] = None,
) -> Result:
    return KnitTree(results, virtual_gates).knit()
