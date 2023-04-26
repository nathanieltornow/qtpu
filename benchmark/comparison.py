import os

from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.providers.fake_provider import FakeBackendV2, FakeGuadalupeV2
from qiskit.providers.ibmq import IBMQ, AccountProvider, IBMQBackend

from circuits 
from fidelity import calc_fidelity
from util import append_to_csv_file


def compare_device() -> IBMQBackend | FakeBackendV2:
    return FakeGuadalupeV2()


def run_comparison(
    circuits: list[QuantumCircuit],
    backend: IBMQBackend | FakeBackendV2,
    provider: AccountProvider,
    csv_name: str,
):
    if isinstance(backend, IBMQBackend):
        n_qubits = backend.configuration().num_qubits
    else:
        n_qubits = backend.num_qubits

    circuits = [circ for circ in circuits if len(circ.qubits) <= n_qubits]
    transpiled_circuits = transpile(circuits, backend=backend, optimization_level=3)
    counts = backend.run(transpiled_circuits, shots=20000).result().get_counts()
    counts = [counts] if isinstance(counts, dict) else counts
    fidelities = [
        calc_fidelity(circ, count, provider) for circ, count in zip(circuits, counts)
    ]

    for circ, fid in zip(circuits, fidelities):
        append_to_csv_file(
            csv_name,
            {
                "num_qubits": len(circ.qubits),
                "fidelity": fid,
            },
        )


def run_ghz_comparison(backend: IBMQBackend | FakeBackendV2, provider: AccountProvider):
    csv_name = f"results/compare/{backend.name}/ghz.csv"

    qasms = os.listdir("qasm/ghz")
    circuits = [
        QuantumCircuit.from_qasm_file("qasm/ghz/" + qasm)
        for qasm in qasms
        if qasm.endswith(".qasm")
    ]
    circuits = [circ for circ in circuits] * 5
    run_comparison(circuits, backend, provider, csv_name)


def run_twolocal_comparison(
    backend: IBMQBackend | FakeBackendV2, provider: AccountProvider, reps: int
):
    csv_name = f"results/compare/{backend.name}/twolocal_{reps}.csv"

    qasms = [f"qasm/twolocal/{reps}_{i}.qasm" for i in range(2, 17, 2)]
    circuits = [QuantumCircuit.from_qasm_file(qasm) for qasm in qasms]
    circuits = [circ for circ in circuits] * 5
    run_comparison(circuits, backend, provider, csv_name)


def run_qft_comparison(backend: IBMQBackend | FakeBackendV2, provider: AccountProvider):
    csv_name = f"results/compare/{backend.name}/qft.csv"

    qasms = os.listdir("qasm/qft")
    circuits = [
        QuantumCircuit.from_qasm_file("qasm/qft/" + qasm)
        for qasm in qasms
        if qasm.endswith(".qasm")
    ]
    circuits = [circ for circ in circuits] * 5
    run_comparison(circuits, backend, provider, csv_name)


def run_dj_comparison(backend: IBMQBackend | FakeBackendV2, provider: AccountProvider):
    csv_name = f"results/compare/{backend.name}/dj.csv"

    qasms = os.listdir("qasm/dj")
    circuits = [
        QuantumCircuit.from_qasm_file("qasm/dj/" + qasm)
        for qasm in qasms
        if qasm.endswith(".qasm")
    ]
    circuits = [circ for circ in circuits] * 5
    run_comparison(circuits, backend, provider, csv_name)


def run_qaoa_comparison(
    backend: IBMQBackend | FakeBackendV2, provider: AccountProvider, graph_type: str
):
    csv_name = f"results/compare/{backend.name}/qaoa_{graph_type}.csv"

    qasms = os.listdir("qasm/qaoa")
    circuits = [
        QuantumCircuit.from_qasm_file("qasm/qaoa/" + qasm)
        for qasm in qasms
        if qasm.endswith(".qasm") and qasm.startswith(graph_type)
    ]
    circuits = [circ for circ in circuits] * 5
    run_comparison(circuits, backend, provider, csv_name)


if __name__ == "__main__":
    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q-research', group='bundeswehr-uni-1', project='main')
    
    backend = provider.get_backend("ibm_oslo")
    # run_ghz_comparison(backend, provider)
    # run_twolocal_comparison(backend, provider, 1)
    # run_twolocal_comparison(backend, provider, 2)

    # run_qaoa_comparison(backend, provider, "l")
    # run_qaoa_comparison(backend, provider, "k")
    run_qaoa_comparison(backend, provider, "k")
