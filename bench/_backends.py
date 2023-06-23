from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session
from qiskit_ibm_runtime.options import (
    Options,
    SimulatorOptions,
    ExecutionOptions,
)
from qiskit_aer.noise import NoiseModel
from qiskit.circuit import QuantumCircuit

from qvm.quasi_distr import QuasiDistr

from qiskit.providers import BackendV2
from qiskit.providers.fake_provider import (
    FakeMumbaiV2,
    FakeHanoiV2,
    FakeCairoV2,
    FakeKolkataV2,
    FakeAuckland,
)
from qiskit_aer import AerSimulator


BACKEND_TYPES = {
    "mumbai": FakeMumbaiV2,
    "hanoi": FakeHanoiV2,
    "cairo": FakeCairoV2,
    "kolkata": FakeKolkataV2,
    "auckland": FakeAuckland,
}


class SimulatorRunner:
    def __init__(self, service: QiskitRuntimeService) -> None:
        self._session = Session(service, backend="ibmq_qasm_simulator")
        self._options = Options(
            optimization_level=0,
            resilience_level=0,
            execution=ExecutionOptions(
                shots=20000,
            ),
        )

    def run(self, circuits: list[QuantumCircuit]) -> list[QuasiDistr]:
        sampler = Sampler(
            session=self._session,
            options=self._options,
        )
        job = sampler.run(circuits)
        return [
            QuasiDistr.from_sampler_distr(dist, circ.num_clbits)
            for circ, dist in zip(circuits, job.result().quasi_dists)
        ]


class SimulatedBackendRunner:
    def __init__(self, service: QiskitRuntimeService, backend: BackendV2) -> None:
        self._backend = backend
        self._session = Session(service, backend="ibmq_qasm_simulator")
        self._options = Options(
            optimization_level=0,
            resilience_level=0,
            execution=ExecutionOptions(
                shots=20000,
            ),
            simulator=SimulatorOptions(
                coupling_map=self._backend.coupling_map,
                noise_model=NoiseModel.from_backend(self._backend),
                basis_gates=["cx", "id", "rz", "sx", "x"],
            ),
        )

    @property
    def backend(self) -> BackendV2:
        return self._backend

    def run(self, circuits: list[QuantumCircuit]) -> list[QuasiDistr]:
        sampler = Sampler(
            session=self._session,
            options=self._options,
        )
        job = sampler.run(circuits)
        return [
            QuasiDistr.from_sampler_distr(dist, circ.num_clbits)
            for circ, dist in zip(circuits, job.result().quasi_dists)
        ]


def generate_default_simulator_options(
    noisy_backend: BackendV2 | None = None,
) -> Options:
    sim_options = None
    if noisy_backend is not None:
        sim_options = SimulatorOptions(
            coupling_map=noisy_backend.coupling_map,
            noise_model=NoiseModel.from_backend(noisy_backend),
            basis_gates=["cx", "id", "rz", "sx", "x"],
        )
    return Options(
        optimization_level=3,
        resilience_level=0,
        execution=ExecutionOptions(
            shots=20000,
        ),
        simulator=sim_options,
    )
