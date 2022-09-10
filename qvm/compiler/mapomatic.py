from typing import List

from qiskit.providers import ProviderV1, Backend
import mapomatic as mm

from qvm.circuit import DistributedCircuit
from .compiler import Compiler


class MapomaticCompiler(Compiler):
    provider: ProviderV1
    backends: List[Backend]

    def __init__(self, provider: ProviderV1) -> None:
        self.provider = provider
        self.backends = provider.backends()

    def run(self, vc: DistributedCircuit) -> DistributedCircuit:
        fragments = vc.fragments
        for frag in fragments:
            frag_circ = vc.fragment_as_circuit(frag)
            best_layout = mm.best_overall_layout(frag_circ, self.backends)
            layout = best_layout[0]
            backend = self.provider.get_backend(best_layout[1])
            vc.map_fragment(frag, backend, layout)
        return vc
