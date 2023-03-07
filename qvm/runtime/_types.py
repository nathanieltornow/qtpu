import abc
from dataclasses import dataclass

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from qvm.quasi_distr import QuasiDistr


@dataclass
class SampleMetaData:
    shots: int = 10000
    initial_layout: list | None = None


class QPU(abc.ABC):
    @abc.abstractmethod
    def sample(
        self, circuits: list[QuantumCircuit], metadata: SampleMetaData
    ) -> list[QuasiDistr]:
        ...

    @abc.abstractmethod
    def name(self) -> str:
        ...

    @abc.abstractmethod
    def num_qubits(self) -> int:
        ...

    @abc.abstractmethod
    def phyisical_layout(self) -> CouplingMap | None:
        ...

    @abc.abstractmethod
    def expected_noise(self) -> float | None:
        ...


# qpu_name -> metadata
RuntimeMetaData = dict[str, SampleMetaData]


class ResourceManager(abc.ABC):
    @abc.abstractmethod
    def sample(
        self, circuits: list[QuantumCircuit], metadata: RuntimeMetaData
    ) -> list[QuasiDistr]:
        ...

    @abc.abstractmethod
    def qpus(self) -> list[QPU]:
        ...
