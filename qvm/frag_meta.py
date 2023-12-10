from qiskit.providers import BackendV2
from qiskit_aer import AerSimulator


class FragmentMetadata:
    def __init__(self, backend: BackendV2 | None = None) -> None:
        self.backend = backend or AerSimulator()
