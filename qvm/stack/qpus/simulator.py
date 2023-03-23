import itertools
from uuid import uuid4

from qiskit_aer.noise import NoiseModel
from qiskit.circuit import QuantumCircuit
from qiskit.providers.ibmq import AccountProvider
from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit.providers.fake_provider import FakeBackendV2
from qiskit.transpiler import CouplingMap
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator, AerJob, StatevectorSimulator

from qvm.stack._types import QPU, insert_placeholders, QernelArgument, QVMJobMetadata
from qvm.quasi_distr import QuasiDistr


class LocalSimulator(QPU):
    def __init__(self) -> None:
        super().__init__()
        self._jobs: dict[str, AerJob] = {}

    def num_qubits(self) -> int:
        return 15

    def coupling_map(self) -> CouplingMap:
        combinations = list(itertools.combinations(range(self.num_qubits()), 2))
        combinations += [(b, a) for a, b in combinations]
        return CouplingMap(combinations)

    def run(
        self,
        qernel: QuantumCircuit,
        args: list[QernelArgument],
        metadata: QVMJobMetadata,
    ) -> str:
        backend = AerSimulator()
        if len(args) == 0:
            circs = [qernel]
        else:
            circs = [insert_placeholders(qernel, arg) for arg in args]
        circs = transpile(circs, backend=backend, optimization_level=0)
        job = backend.run(circs, shots=metadata.shots)
        job_id = str(uuid4())
        self._jobs[job_id] = job
        return job_id

    def get_results(self, job_id: str) -> list[QuasiDistr]:
        if job_id not in self._jobs:
            raise ValueError("Job ID not found")
        counts = self._jobs[job_id].result().get_counts()
        counts = [counts] if isinstance(counts, dict) else counts
        return [QuasiDistr.from_counts(count) for count in counts]


class IBMQSimulator(QPU):
    def __init__(
        self, provider: AccountProvider, fake_backend: FakeBackendV2 | None = None
    ) -> None:
        super().__init__()
        self._backend = provider.get_backend("ibmq_qasm_simulator")
        self._fake_backend = fake_backend
        self._jobsets: dict[str, tuple[str, int]] = {}

    def num_qubits(self) -> int:
        if self._fake_backend is not None:
            return self._fake_backend.num_qubits
        return 32

    def coupling_map(self) -> CouplingMap:
        if self._fake_backend is not None:
            return self._fake_backend.coupling_map
        combinations = list(itertools.combinations(range(self.num_qubits()), 2))
        combinations += [(b, a) for a, b in combinations]
        return CouplingMap(combinations)

    def run(
        self,
        qernel: QuantumCircuit,
        args: list[QernelArgument],
        metadata: QVMJobMetadata,
    ) -> str:
        if self._fake_backend is not None:
            qernel = transpile(qernel, backend=self._fake_backend, optimization_level=3)
        if len(args) == 0:
            circs = [qernel]
        else:
            circs = [insert_placeholders(qernel, arg) for arg in args]
        circs = transpile(circs, backend=self._backend, optimization_level=0)

        noise_model, coupling_map = None, None

        if self._fake_backend is not None:
            noise_model = NoiseModel.from_backend(self._fake_backend)

        job_manager = IBMQJobManager()
        job_set = job_manager.run(
            circs,
            backend=self._backend,
            shots=metadata.shots,
            noise_model=noise_model,
        )
        job_set_id = job_set.job_set_id()
        job_id = str(uuid4())
        self._jobsets[job_id] = (job_set_id, len(circs))
        return job_id

    def get_results(self, job_id: str) -> list[QuasiDistr]:
        if job_id not in self._jobsets:
            raise ValueError(f"No job with id {job_id}")

        job_set_results = (
            IBMQJobManager()
            .retrieve_job_set(
                self._jobsets[job_id][0],
                provider=self._backend.provider(),
                refresh=True,
            )
            .results()
        )
        counts = [
            job_set_results.get_counts(i) for i in range(self._jobsets[job_id][1])
        ]
        return [QuasiDistr.from_counts(counts[i]) for i in range(len(counts))]
