import abc
from dataclasses import dataclass
from multiprocessing.pool import Pool
from time import perf_counter, time
from uuid import uuid4

from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.compiler import transpile

from qvm.cut_library.decomposition import bisect, bisect_recursive, decompose
from qvm.quasi_distr import QuasiDistr
from qvm.stack._types import QPU, QernelArgument, QVMJobMetadata, QVMLayer
from qvm.stack._virtualizer import Virtualizer
from qvm.virtual_gates import VirtualBinaryGate


@dataclass
class Stat:
    cut_time: float = -1.0
    exec_start: float = -1.0
    exec_time: float = -1.0
    knit_time: float = -1.0
    num_vgates: int = -1


class Decomposer(QVMLayer, abc.ABC):
    def __init__(self, sub_layer: QVMLayer) -> None:
        super().__init__()
        self._sub_layer = sub_layer
        self._virtualizers: dict[str, Virtualizer] = {}
        self._sub_jobs: dict[str, dict[QuantumRegister, str]] = {}
        self._stats: dict[str, Stat] = {}

    def qpus(self) -> dict[str, QPU]:
        return self._sub_layer.qpus()

    @abc.abstractmethod
    def _decompose(self, qernel: QuantumCircuit) -> QuantumCircuit:
        ...

    def run(
        self,
        qernel: QuantumCircuit,
        args: list[QernelArgument],
        metadata: QVMJobMetadata,
    ) -> str:
        assert len(args) == 0

        qernel = transpile(
            qernel,
            optimization_level=3,
        ).decompose()

        print("Decomposing qernel")

        job_stat = Stat()
        now = perf_counter()
        qernel = self._decompose(qernel)
        job_stat.cut_time = perf_counter() - now

        job_stat.num_vgates = sum(
            1
            for cinstr in qernel.data
            if isinstance(cinstr.operation, VirtualBinaryGate)
        )

        print(f"Decomposed qernel into {job_stat.num_vgates} virtual gates")
        print(f"Decomposed qernel in {job_stat.cut_time} seconds")

        job_stat.exec_start = time()
        virtualizer = Virtualizer(qernel)
        sub_jobs = {}
        for qreg, sub_qernel in virtualizer.sub_qernels().items():
            instantiations = virtualizer.instantiations(qreg)
            print(
                f"Submitting fragment {qreg.name} with {len(sub_qernel.qubits)} qubits and {len(instantiations)} instantiations"
            )
            sub_job_id = self._sub_layer.run(sub_qernel, instantiations, metadata)
            sub_jobs[qreg] = sub_job_id

        job_id = str(uuid4())
        self._virtualizers = {job_id: virtualizer}
        self._sub_jobs[job_id] = sub_jobs
        self._stats[job_id] = job_stat
        return job_id

    def get_results(self, job_id: str) -> list[QuasiDistr]:
        if job_id not in self._sub_jobs:
            raise ValueError(f"Job {job_id} not found")
        job_stats = self._stats[job_id]
        sub_jobs = self._sub_jobs[job_id]
        if len(sub_jobs) == 1:
            return self._sub_layer.get_results(list(sub_jobs.values())[0])
        
        virtualizer = self._virtualizers[job_id]
        for qreg, sub_job_id in sub_jobs.items():
            sub_results = self._sub_layer.get_results(sub_job_id)
            virtualizer.put_results(qreg, sub_results)
        job_stats.exec_time = time() - job_stats.exec_start

        print("Knitting results")
        now = perf_counter()
        res = [virtualizer.knit()]
        job_stats.knit_time = perf_counter() - now
        print(f"Knitted results in {job_stats.knit_time} seconds")
        return res
