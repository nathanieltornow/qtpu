from qvm.runtime.types import QVMInterface, QPU


class RessourceManager(QVMInterface):
    def __init__(self, qpus: set[QPU]) -> None:
        super().__init__()
        self._qpus = qpus

    