from uuid import uuid4

from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.providers import BackendV1 as QPU

from qvm.cut_library.decomposition import bisect
from qvm.quasi_distr import QuasiDistr

from ._types import QernelArgument, QVMJobMetadata, QVMLayer
from ._virtualizer import Virtualizer


class Decomposer(QVMLayer):
    def __init__(self, sub_layer: QVMLayer) -> None:
        super().__init__()
        self._sub_layer = sub_layer
        self._virtualizers: dict[str, Virtualizer] = {}
        self._sub_jobs: dict[str, dict[QuantumRegister, str]] = {}

    def qpus(self) -> dict[str, QPU]:
        return self._sub_layer.qpus()

    def run(
        self,
        qernel: QuantumCircuit,
        args: list[QernelArgument],
        metadata: QVMJobMetadata,
    ) -> str:
        assert len(args) == 0

        qernel = bisect(qernel)

        virtualizer = Virtualizer(qernel)
        sub_jobs = {}
        for qreg, sub_qernel in virtualizer.sub_qernels().items():
            instantiations = virtualizer.instantiations(qreg)
            sub_job_id = self._sub_layer.run(sub_qernel, instantiations, metadata)
            sub_jobs[qreg] = sub_job_id

        job_id = str(uuid4())
        self._virtualizers = {job_id: virtualizer}
        self._sub_jobs = {job_id: sub_jobs}
        return job_id

    def get_results(self, job_id: str) -> list[QuasiDistr]:
        if job_id not in self._sub_jobs:
            raise ValueError("Job not found")
        sub_jobs = self._sub_jobs[job_id]
        virtualizer = self._virtualizers[job_id]
        for qreg, sub_job_id in sub_jobs.items():
            sub_results = self._sub_layer.get_results(sub_job_id)
            virtualizer.put_results(qreg, sub_results)
        return [virtualizer.knit()]
