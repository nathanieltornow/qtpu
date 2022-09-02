from typing import Any, Dict, Optional
from qiskit import transpile
from qiskit.providers import Backend
from qvm.transpiler.fragmented_circuit import FragmentedCircuit

from qvm.transpiler.transpiler import DistributedPass


class SingleDeviceMapping(DistributedPass):
    def __init__(
        self, backend: Backend, transpile_flags: Optional[Dict[str, Any]] = None
    ) -> None:
        self.backend = backend
        if transpile_flags is None:
            transpile_flags = {"optimization_level": 2}
        self.transpile_flags = transpile_flags

    def run(self, frag_circ: FragmentedCircuit) -> None:
        fragments = frag_circ.fragments
        for frag in fragments:
            t_circ = transpile(
                frag, self.backend, **self.transpile_flags
            )
            frag_circ.replace_fragment(frag, t_circ, self.backend)
