from qiskit import QuantumCircuit
from qiskit.providers.ibmq import AccountProvider
from qiskit.quantum_info import hellinger_fidelity
from qiskit.compiler import transpile
from qiskit_aer import StatevectorSimulator


def perfect_counts(
    original_circuit: QuantumCircuit, provider: AccountProvider | None = None
) -> dict[str, int]:
    if provider is not None:
        print("starting job")
        backend = provider.get_backend("simulator_statevector")
        circ = transpile(original_circuit, backend=backend, optimization_level=0)
        cnt = backend.run(circ, shots=100000).result().get_counts()
        print("finished job")
    else:
        cnt = (
            StatevectorSimulator()
            .run(original_circuit, shots=100000)
            .result()
            .get_counts()
        )
    return {k.replace(" ", ""): v for k, v in cnt.items()}


def fidelity(
    orginal_circuit: QuantumCircuit,
    noisy_counts: dict[str, int],
    provider: AccountProvider | None = None,
) -> float:
    return hellinger_fidelity(perfect_counts(orginal_circuit, provider), noisy_counts)
