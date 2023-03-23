from qiskit.circuit import QuantumCircuit
from qiskit.providers import Job
from qiskit.providers.fake_provider import FakeBackend, FakeBackendV2
from qiskit.providers.ibmq import AccountProvider
from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit_aer.noise import NoiseModel

from qvm.quasi_distr import QuasiDistr


def run_fake_backend_on_ibmq(
    circuits: list[QuantumCircuit],
    backend: FakeBackend | FakeBackendV2,
    provider: AccountProvider,
    shots: int,
) -> list[QuasiDistr]:
    noise_model = NoiseModel.from_backend(backend)
    if isinstance(backend, FakeBackend):
        coupling_map = backend.configuration().coupling_map
    else:
        coupling_map = backend.coupling_map
    basis_gates = noise_model.basis_gates
    
    backend = provider.get_backend("ibmq_qasm_simulator")
    results = IBMQJobManager().run(circuits, backend, shots=shots, noise_model=noise_model, coupling_map=coupling_map, basis_gates=basis_gates).results()
    counts = [results().get_counts(i) for i in range(len(circuits))]
    return [QuasiDistr.from_counts(count, shots) for count in counts]