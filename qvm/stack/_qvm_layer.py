import abc
from dataclasses import dataclass

from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV1 as QPU

from qvm.quasi_distr import QuasiDistr

from ._types import QernelArgument


@dataclass
class QVMJobMetadata:
    qpus: list[QPU] | None = None
    initial_layout: list[int] | None = None
    shots: int = 10000


class QVMLayer(abc.ABC):
    @abc.abstractmethod
    def qpus(self) -> list[QPU]:
        ...

    @abc.abstractmethod
    def run(
        self,
        qernel: QuantumCircuit,
        args: list[QernelArgument],
        metadata: QVMJobMetadata,
    ) -> str:
        ...

    @abc.abstractmethod
    def get_results(self, job_id: str) -> list[QuasiDistr]:
        ...
