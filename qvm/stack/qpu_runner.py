from uuid import uuid4

from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.providers import BackendV1 as QPU
from qiskit.providers import Job

from qvm.quasi_distr import QuasiDistr

from ._types import (QernelArgument, QVMJobMetadata, QVMLayer,
                     insert_placeholders)


class QPURunner(QVMLayer):
    def __init__(self, qpus: dict[str, QPU]) -> None:
        super().__init__()
        self._qpus = qpus
        self._jobs: dict[str, Job] = {}

    def qpus(self) -> dict[str, QPU]:
        return self._qpus.copy()

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
        if metadata.qpu_name not in self._qpus:
            raise ValueError(f"No QPU with name {metadata.qpu_name}")
        
        print("hi")
        qpu = self._qpus[metadata.qpu_name]
        transpiled_qernel = transpile(
            qernel,
            backend=qpu,
            initial_layout=metadata.initial_layout,
            optimization_level=3,
        )
        circs = [insert_placeholders(transpiled_qernel, arg) for arg in args]
        circs = transpile(
            circs,
            backend=qpu,
            initial_layout=metadata.initial_layout,
            optimization_level=0,
        )
        job = qpu.run(circs, shots=metadata.shots)
        job_id = str(uuid4())
        self._jobs[job_id] = job
        return job_id

    def get_results(self, job_id: str) -> list[QuasiDistr]:
        if job_id not in self._jobs:
            raise ValueError(f"No job with id {job_id}")
        job = self._jobs[job_id]
        counts = job.result().get_counts()
        counts = [counts] if isinstance(counts, dict) else counts
        return [QuasiDistr.from_counts(c) for c in counts]
