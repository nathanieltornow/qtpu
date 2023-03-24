from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.providers.ibmq import AccountProvider
from qiskit.quantum_info import hellinger_fidelity
from qiskit_aer import StatevectorSimulator


def merge_counts(counts: list[dict[str, int]]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for cnt in counts:
        for k, v in cnt.items():
            merged[k] = merged.get(k, 0) + v
    return merged


def perfect_counts(
    original_circuit: QuantumCircuit, provider: AccountProvider
) -> dict[str, int]:
    backend = provider.get_backend("simulator_statevector")
    circ = transpile(original_circuit, backend=backend, optimization_level=0)
    cnt = backend.run([circ]*10, shots=100000).result().get_counts()
    cnt = merge_counts(cnt)
    return {k.replace(" ", ""): v for k, v in cnt.items()}


def calc_fidelity(
    orginal_circuit: QuantumCircuit,
    noisy_counts: dict[str, int],
    provider: AccountProvider,
) -> float:
    return hellinger_fidelity(perfect_counts(orginal_circuit, provider), noisy_counts)
