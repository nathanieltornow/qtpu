import itertools
from uuid import uuid4

from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.providers.ibmq import AccountProvider
from qiskit.providers.ibmq.managed import IBMQJobManager, ManagedJobSet
from qiskit.transpiler import CouplingMap

from qvm.quasi_distr import QuasiDistr
from qvm.stack._types import (
    QPU,
    QernelArgument,
    QVMJobMetadata,
    insert_placeholders,
    PlaceholderGate,
)


class IBMQQPU(QPU):
    def __init__(self, provider: AccountProvider, backend_name: str) -> None:
        super().__init__()
        self._backend = provider.get_backend(backend_name)
        self._jobsets: dict[str, tuple[ManagedJobSet, int]] = {}

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
        qernel = transpile(
            qernel,
            backend=self._backend,
            initial_layout=metadata.initial_layout,
            optimization_level=3,
        )
        if len(args) == 0:
            circs = [qernel]
        else:
            circs = [insert_placeholders(qernel, arg) for arg in args]
        circs = transpile(
            circs,
            backend=self._backend,
            initial_layout=metadata.initial_layout,
            optimization_level=0,
        )
        job_id = str(uuid4())
        if len(circs) == 1:
            circs = circs[0]
        job_manager = IBMQJobManager()
        job_set = job_manager.run(circs, backend=self._backend, shots=metadata.shots)
        self._jobsets[job_id] = (job_set, len(circs))
        return job_id

    def get_results(self, job_id: str) -> list[QuasiDistr]:
        if job_id not in self._jobsets:
            raise ValueError(f"No job with id {job_id}")
        job_set_results = self._jobsets[job_id][0].results()

        counts = [
            job_set_results.get_counts(i) for i in range(self._jobsets[job_id][1])
        ]
        return [QuasiDistr.from_counts(counts[i]) for i in range(len(counts))]
