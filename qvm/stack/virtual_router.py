from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV1 as QPU

from qvm.quasi_distr import QuasiDistr

from ._types import QernelArgument, QVMJobMetadata, QVMLayer
from .qpu_runner import QPURunner


class _VirtualRouterJob:
    pass


class VirtualRouter(QVMLayer):
    def __init__(self, runner: QPURunner) -> None:
        super().__init__()
        self._runner = runner
        self._jobs: dict[str, str] = {}

    def qpus(self) -> dict[str, QPU]:
        return self._runner.qpus()

    def run(
        self,
        qernel: QuantumCircuit,
        args: list[QernelArgument],
        metadata: QVMJobMetadata,
    ) -> str:
        if len(args) == 0:
            raise ValueError("No arguments specified")
        if not metadata.qpu_name:
            raise ValueError("No QPU specified")
        if metadata.vgates_to_spend <= 0:
            return self._runner.run(qernel, args, metadata)
        return super().run(qernel, args, metadata)

    def get_results(self, job_id: str) -> list[QuasiDistr]:
        pass
