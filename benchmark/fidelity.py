from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.providers.ibmq import AccountProvider
from qiskit.quantum_info import hellinger_fidelity
from qiskit_aer import StatevectorSimulator


def perfect_counts(
    original_circuit: QuantumCircuit, provider: AccountProvider
) -> dict[str, int]:
    backend = StatevectorSimulator()
    circ = transpile(original_circuit, backend=backend, optimization_level=0)
    cnt = backend.run(circ, shots=20000).result().get_counts()
    return {k.replace(" ", ""): v for k, v in cnt.items()}


def calc_fidelity(
    orginal_circuit: QuantumCircuit,
    noisy_counts: dict[str, int],
    provider: AccountProvider,
) -> float:
    return hellinger_fidelity(perfect_counts(orginal_circuit, provider), noisy_counts)
