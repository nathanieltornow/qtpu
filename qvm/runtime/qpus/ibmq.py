import itertools
from uuid import uuid4

from qiskit.circuit import QuantumCircuit
from qiskit.providers.ibmq import AccountProvider
from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit.transpiler import CouplingMap
from qiskit_aer.noise import NoiseModel

from qvm.quasi_distr import QuasiDistr
from qvm.types import QPU, SampleMetaData


class IBMQQPU(QPU):
    def __init__(self, backend_name: str, provider: AccountProvider) -> None:
        super().__init__()
        self._name = backend_name
        self._backend = provider.get_backend(backend_name)

    def name(self) -> str:
        return self._name


class IBMQSimulator(IBMQQPU):
    def __init__(
        self,
        provider: AccountProvider,
        noise_model: NoiseModel | None = None,
        coupling_map: CouplingMap | None = None,
    ) -> None:
        super().__init__(backend_name="ibmq_qasm_simulator", provider=provider)
        self._name = self._name + str(uuid4())
        self._noise_model = noise_model
        self._coupling_map = coupling_map

    def sample(
        self, circuits: list[QuantumCircuit], metadata: SampleMetaData
    ) -> list[QuasiDistr]:
        manager = IBMQJobManager()
        results = manager.run(
            circuits,
            backend=self._backend,
            shots=metadata.shots,
            noise_model=self._noise_model,
            coupling_map=self._coupling_map,
        ).results()
        counts = [results.get_counts(i) for i in range(len(circuits))]
        return [
            QuasiDistr.from_counts(counts=count, shots=metadata.shots)
            for count in counts
        ]

    def num_qubits(self) -> int:
        return 32

    def phyisical_layout(self) -> CouplingMap | None:
        return self._coupling_map

    def expected_noise(self, circuit: QuantumCircuit) -> float | None:
        return 0.0
