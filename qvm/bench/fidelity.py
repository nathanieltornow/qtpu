from qiskit import QuantumCircuit
from qiskit.providers.ibmq import AccountProvider
from qiskit.quantum_info import hellinger_fidelity
from qiskit_aer import AerSimulator


def perfect_counts(
    original_circuit: QuantumCircuit, shots: int, provider: AccountProvider | None = None
) -> dict[str, int]:
    if provider is not None:
        cnt = (
            provider.get_backend("simulator_mps")
            .run(original_circuit, shots=shots)
            .result()
            .get_counts()
        )
    else:
        cnt = (
            AerSimulator()
            .run(original_circuit, shots=shots)
            .result()
            .get_counts()
        )
    return {k.replace(" ", ""): v for k, v in cnt.items()}


def fidelity(
    orginal_circuit: QuantumCircuit,
    noisy_counts: dict[str, int],
    provider: AccountProvider | None = None,
) -> float:
    return hellinger_fidelity(perfect_counts(orginal_circuit, sum(noisy_counts.values()), provider), noisy_counts)
