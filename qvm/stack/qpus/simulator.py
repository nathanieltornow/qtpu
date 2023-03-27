import itertools
from uuid import uuid4

from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.providers.ibmq import AccountProvider
from qiskit.transpiler import CouplingMap
from qiskit.providers.fake_provider import FakeBackendV2
from qiskit_aer import AerJob, AerSimulator

from qvm.quasi_distr import QuasiDistr
from qvm.stack._types import QPU, QernelArgument, insert_placeholders, QVMJobMetadata


class SimulatorQPU(QPU):
    def __init__(self, fake_backend: FakeBackendV2 | None = None) -> None:
        super().__init__()
        self._backend = fake_backend or AerSimulator()
        self._jobsets: dict[str, AerJob] = {}
        self._num_qubits = fake_backend.num_qubits if fake_backend else 15

    def num_qubits(self) -> int:
        return self._num_qubits

    def coupling_map(self) -> CouplingMap:
        return CouplingMap.from_full(self._num_qubits)

    def run(
        self,
        qernel: QuantumCircuit,
        args: list[QernelArgument],
        metadata: QVMJobMetadata,
    ) -> str:
        if len(args) == 0:
            circs = [qernel]
        else:
            circs = [insert_placeholders(qernel, arg) for arg in args]
        circs = transpile(
            circs,
            backend=self._backend,
            optimization_level=0,
        )

        job_id = str(uuid4())
        job = self._backend.run(circs, shots=metadata.shots)
        self._jobsets[job_id] = job
        return job_id

    def get_results(self, job_id: str) -> list[QuasiDistr]:
        if job_id not in self._jobsets:
            raise ValueError(f"No job with id {job_id}")
        job = self._jobsets[job_id]
        counts = job.result().get_counts()
        counts = [counts] if isinstance(counts, dict) else counts
        return [QuasiDistr.from_counts(count) for count in counts]
