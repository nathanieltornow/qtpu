import abc

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from qvm.core.types import Argument
from qvm.core.quasi_distr import QuasiDistr


class QVMInterface(abc.ABC):
    @abc.abstractmethod
    def _run(
        self,
        circuit: QuantumCircuit,
        args: list[Argument],
        shots: int = 20000,
        max_overhead: int = 300,
    ) -> str:
        ...

    def run(self, circuit: QuantumCircuit, shots: int = 20000) -> str:
        args: list[Argument] = [Argument()]
        return self._run(circuit, args, shots)

    @abc.abstractmethod
    def results(self, job_id: str) -> list[QuasiDistr]:
        ...


class QPU(QVMInterface, abc.ABC):
    @abc.abstractmethod
    def coupling_map(self) -> CouplingMap:
        ...

    @abc.abstractmethod
    def num_qubits(self) -> int:
        ...
