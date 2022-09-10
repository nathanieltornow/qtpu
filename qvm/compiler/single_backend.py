from qiskit.providers import Backend

from qvm.circuit import DistributedCircuit
from .compiler import Compiler


class SingleBackend(Compiler):
    def __init__(self, backend: Backend) -> None:
        self.backend = backend

    def run(self, vc: DistributedCircuit) -> DistributedCircuit:
        fragments = vc.fragments
        for frag in fragments:
            vc.map_fragment(frag, self.backend)
        return vc
