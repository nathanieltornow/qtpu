from uuid import uuid4
from time import time, perf_counter

# from multiprocessing.pool import Pool

from qiskit.circuit import QuantumCircuit, QuantumRegister

from qvm.cut_library.decomposition import decompose
from qvm.quasi_distr import QuasiDistr

from ._types import QernelArgument, QVMJobMetadata, QVMLayer, QPU
from ._virtualizer import Virtualizer


class Decomposer(QVMLayer):
    def __init__(self, sub_layer: QVMLayer, max_qpu_utilization: int = 2) -> None:
        super().__init__()
        self._sub_layer = sub_layer
        self._virtualizers: dict[str, Virtualizer] = {}
        self._sub_jobs: dict[str, dict[QuantumRegister, str]] = {}
        self._max_qpu_utilization = max_qpu_utilization
        self._pool = None
        self._stats: dict[str, dict[str, float | int]] = {}

    def qpus(self) -> dict[str, QPU]:
        return self._sub_layer.qpus()

    def run(
        self,
        qernel: QuantumCircuit,
        args: list[QernelArgument],
        metadata: QVMJobMetadata,
    ) -> str:
        assert len(args) == 0
        job_stats: dict[str, float | int] = {}

        now = perf_counter()
        if metadata.qpu_name is not None:
            qernel = decompose(
                qernel,
                int(
                    self.qpus()[metadata.qpu_name].num_qubits()
                    / self._max_qpu_utilization
                ),
            )
        else:
            max_qpu_size = max(qpu.num_qubits() for qpu in self.qpus().values())
            qernel = decompose(qernel, int(max_qpu_size / self._max_qpu_utilization))

        job_stats["cut_time"] = perf_counter() - now

        job_stats["exec_start"] = time()
        virtualizer = Virtualizer(qernel)
        sub_jobs = {}
        for qreg, sub_qernel in virtualizer.sub_qernels().items():
            print(
                f"Submitting fragment {qreg.name} with {len(sub_qernel.qubits)} qubits"
            )
            instantiations = virtualizer.instantiations(qreg)
            sub_job_id = self._sub_layer.run(sub_qernel, instantiations, metadata)
            sub_jobs[qreg] = sub_job_id

        job_id = str(uuid4())
        self._virtualizers = {job_id: virtualizer}
        self._sub_jobs = {job_id: sub_jobs}
        self._stats[job_id] = job_stats
        return job_id

    def get_results(self, job_id: str) -> list[QuasiDistr]:
        if job_id not in self._sub_jobs:
            raise ValueError("Job not found")
        job_stats = self._stats[job_id]
        sub_jobs = self._sub_jobs[job_id]
        virtualizer = self._virtualizers[job_id]
        for qreg, sub_job_id in sub_jobs.items():
            sub_results = self._sub_layer.get_results(sub_job_id)
            virtualizer.put_results(qreg, sub_results)
        job_stats["exec_time"] = time() - job_stats["exec_start"]

        now = perf_counter()
        res = [virtualizer.knit(self._pool)]
        job_stats["knit_time"] = perf_counter() - now
        return res
