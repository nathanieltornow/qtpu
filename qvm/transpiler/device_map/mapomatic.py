from typing import Any, Dict, List, Optional

from qiskit.providers import Backend

from qvm.transpiler.fragmented_circuit import FragmentedCircuit
from qvm.transpiler.transpiler import DistributedPass


class MapomaticMapping(DistributedPass):
    def __init__(
        self,
        backends: List[Backend],
        transpile_flags: Optional[Dict[str, Any]],
    ) -> None:
        if len(backends) == 0:
            raise ValueError("No backends provided")
        self.backends = backends
        if transpile_flags is None:
            transpile_flags = {"optimization_level": 3}
        self.transpile_flags = transpile_flags

    def run(self, frag_circ: FragmentedCircuit) -> None:
        pass
