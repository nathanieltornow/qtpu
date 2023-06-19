import abc

from qiskit.providers import BackendV2

from qvm.dag import DAG


class QVMPass(abc.ABC):
    @abc.abstractmethod
    def run(self, dag: DAG, virt_budget: int):
        ...


class ForEachFragmentPass(abc.ABC):
    def __init__(self, passes: list[QVMPass]):
        self._passes = passes
