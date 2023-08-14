import abc
from uuid import uuid4

import mapomatic as mm
from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.primitives import BackendSampler
from qiskit.providers import BackendV2
from qiskit_ibm_runtime import (
    QiskitRuntimeService,
    Session,
    Sampler,
    IBMBackend,
    RuntimeJob,
)
from qiskit_ibm_runtime.options import Options, SimulatorOptions, ExecutionOptions
from qiskit_aer import AerSimulator, AerJob
from qiskit_aer.noise import NoiseModel

from qvm.compiler.dag import DAG
from qvm.quasi_distr import QuasiDistr


DEFAULT_SHOTS = 20000


class QVMBackendRunner(abc.ABC):
    @abc.abstractmethod
    def run(
        self, circuits: list[QuantumCircuit], backend: BackendV2 | None = None
    ) -> str:
        ...

    @abc.abstractmethod
    def get_results(self, job_id: str) -> list[QuasiDistr]:
        ...


class IBMBackendRunner(QVMBackendRunner):
    def __init__(self, provider, simulate_qpus: bool = True) -> None:
        self.simulate_qpus = simulate_qpus
        self._sim = provider.get_backend("ibmq_qasm_simulator")

        # self._qpu_sessions: dict[str, Session] = {}
        # self._simulat )
        self._jobs: dict[str, RuntimeJob] = {}

    def _options(self, noisy_backend: BackendV2 | None = None) -> Options:
        sim_options = None
        if noisy_backend is not None:
            sim_options = SimulatorOptions(
                coupling_map=list(noisy_backend.coupling_map.get_edges()),
                noise_model=NoiseModel.from_backend(noisy_backend),
                basis_gates=["cx", "id", "rz", "sx", "x"],
            )
        return Options(
            optimization_level=0,
            resilience_level=0,
            execution=ExecutionOptions(
                shots=DEFAULT_SHOTS,
            ),
            simulator=sim_options,
        )

    # def _sampler_from_backend(self, backend: BackendV2 | None = None) -> Sampler:
    #     # if backend is None:
    #     #     return Sampler(session=self._simulator_session, options=self._options())
    #     if isinstance(backend, IBMBackend):
    #         if self.simulate_qpus:
    #             return Sampler(
    #                 session=self._simulator_session, options=self._options(backend)
    #             )
    #         if backend.name not in self._qpu_sessions:
    #             self._qpu_sessions[backend.name] = Session(
    #                 service=self._service, backend=backend.name
    #             )
    #         return Sampler(
    #             session=self._qpu_sessions[backend.name], options=self._options()
    #         )
    #     if isinstance(backend, BackendV2):
    #         return Sampler(
    #             session=self._simulator_session, options=self._options(backend)
    #         )
    #     else:
    #         raise TypeError(f"Unknown backend type: {type(backend)}")

    def run(
        self,
        circuits: list[QuantumCircuit],
        backend: BackendV2 | None = None,
    ) -> str:
        if backend is not None:
            circuits = [transpile_circuit(circ, backend) for circ in circuits]
            circuits = [circuits] if isinstance(circuits, QuantumCircuit) else circuits
        else:
            backend = self._sim

        job = backend.run(circuits, shots=DEFAULT_SHOTS, dynamic=True)
        job_id = str(uuid4())
        self._jobs[job_id] = job
        return job_id

    def get_results(self, job_id: str) -> list[QuasiDistr]:
        job = self._jobs[job_id]
        counts = job.result().get_counts()
        counts = [counts] if isinstance(counts, dict) else counts
        return [QuasiDistr.from_counts(count) for count in counts]


class LocalBackendRunner(QVMBackendRunner):
    def __init__(self) -> None:
        self._simulator = AerSimulator()
        self._jobs: dict[str, AerJob] = {}
        super().__init__()

    def run(
        self,
        circuits: list[QuantumCircuit],
        backend: BackendV2 | None = None,
    ) -> str:
        noise_model, coupling_map, basis_gates = None, None, None
        if backend is not None:
            circuits = [transpile_circuit(circ, backend) for circ in circuits]
            coupling_map = backend.coupling_map.get_edges()
            noise_model = NoiseModel.from_backend(backend)
            basis_gates = ["cx", "id", "rz", "sx", "x"]

        job = self._simulator.run(
            circuits,
            shots=DEFAULT_SHOTS,
            baisis_gates=basis_gates,
            noise_model=noise_model,
            coupling_map=coupling_map,
        )
        job_id = str(uuid4())
        self._jobs[job_id] = job
        return job_id

    def get_results(self, job_id: str) -> list[QuasiDistr]:
        job = self._jobs[job_id]
        counts = job.result().get_counts()
        counts = [counts] if isinstance(counts, dict) else counts
        return [QuasiDistr.from_counts(count) for count in counts]


def transpile_circuit(circuit: QuantumCircuit, backend: BackendV2) -> QuantumCircuit:
    t_circ = transpile(circuit, backend=backend, optimization_level=3)
    try:
        dag = DAG(t_circ)
        dag.compact()
        small_qc = dag.to_circuit()
        layouts = mm.matching_layouts(small_qc, backend)
        best_score = mm.evaluate_layouts(small_qc, layouts, backend)
        best_layout = best_score[0][0]
        if best_score[0][1] >= 1.0:
            best_layout = None
    except AttributeError:
        return t_circ
    return transpile(
        small_qc, backend=backend, optimization_level=3, initial_layout=best_layout
    )
