import random
from uuid import uuid4

from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.providers import Job

from qvm.quasi_distr import QuasiDistr

from ._types import QPU, QernelArgument, QVMJobMetadata, QVMLayer, insert_placeholders


class QPURunner(QVMLayer):
    def __init__(self, qpus: dict[str, QPU]) -> None:
        super().__init__()
        self._qpus = qpus
        # job_id -> (qpu_name, qpu_job_id)
        self._jobs: dict[str, tuple[str, str]] = {}

    def qpus(self) -> dict[str, QPU]:
        return self._qpus.copy()

    def run(
        self,
        qernel: QuantumCircuit,
        args: list[QernelArgument],
        metadata: QVMJobMetadata,
    ) -> str:
        if metadata.qpu_name is None:
            qpu_name, qpu = random.choice(list(self._qpus.items()))
        else:
            qpu = self._qpus[metadata.qpu_name]

        qpu_job_id = qpu.run(qernel, args, metadata)
        job_id = str(uuid4())
        self._jobs[job_id] = (qpu_name, qpu_job_id)
        return job_id

    def get_results(self, job_id: str) -> list[QuasiDistr]:
        if job_id not in self._jobs:
            raise ValueError(f"No job with id {job_id}")
        qpu_name, qpu_job_id = self._jobs[job_id]
        qpu = self._qpus[qpu_name]
        return qpu.get_results(qpu_job_id)
