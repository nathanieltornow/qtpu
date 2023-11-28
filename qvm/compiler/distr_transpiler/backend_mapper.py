from qiskit.providers import BackendV2

from qvm.compiler.types import DistributedTranspilerPass
from qvm.virtual_circuit import VirtualCircuit


class BasicBackendMapper(DistributedTranspilerPass):
    def __init__(self, backend: BackendV2) -> None:
        self._backend = backend
        super().__init__()

    def run(self, virt: VirtualCircuit) -> None:
        for frag in virt.fragment_circuits.keys():
            virt.metadata[frag].backend = self._backend
