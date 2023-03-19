


class Mapper:
    
    def __init__(self, qpus: list[QPU]) -> None:
        self._qpus = qpus

    def qpus(self) -> list[QPU]:
        return self._qpus.copy()

    def run(self, qernel: QuantumCircuit, args: list[QernelArgument]) -> str:
        ...