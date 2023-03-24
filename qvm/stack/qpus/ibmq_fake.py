import itertools
from uuid import uuid4

from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.providers.ibmq import AccountProvider
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerJob, AerSimulator

from qvm.quasi_distr import QuasiDistr
from qvm.stack._types import QPU, QernelArgument, insert_placeholders, QVMJobMetadata


class IBMQFakeQPU(QPU):
    def __init__(self, provider: AccountProvider, backend_name: str) -> None:
        super().__init__()
        self._backend = provider.get_backend(backend_name)
        self._simulator = AerSimulator.from_backend(self._backend)
        self._jobsets: dict[str, AerJob] = {}

    def num_qubits(self) -> int:
        return self._backend.configuration().n_qubits

    def coupling_map(self) -> CouplingMap:
        return self._backend.configuration().coupling_map

    def run(
        self,
        qernel: QuantumCircuit,
        args: list[QernelArgument],
        metadata: QVMJobMetadata,
    ) -> str:
        qernel = transpile(qernel, backend=self._backend, optimization_level=3)
        if len(args) == 0:
            circs = [qernel]
        else:
            circs = [insert_placeholders(qernel, arg) for arg in args]
        circs = transpile(circs, backend=self._backend, optimization_level=0)

        job_id = str(uuid4())
        job = self._simulator.run(circs, shots=metadata.shots)
        self._jobsets[job_id] = job
        return job_id

    def get_results(self, job_id: str) -> list[QuasiDistr]:
        if job_id not in self._jobsets:
            raise ValueError(f"No job with id {job_id}")
        job = self._jobsets[job_id]
        counts = job.result().get_counts()
        counts = [counts] if isinstance(counts, dict) else counts
        return [QuasiDistr.from_counts(count) for count in counts]
