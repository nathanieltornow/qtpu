
from qiskit.providers import Backend


class SingleIBMQBackendRuntime:
    
    def __init__(self, backend: BackendV2):
        self._backend = backend