import csv
import os
import sys
from time import perf_counter

from fidelity import calc_fidelity
from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.providers.fake_provider import FakeMumbaiV2
from qiskit.providers.ibmq import IBMQ, AccountProvider
from qiskit_aer.noise import NoiseModel

from qvm.stack._types import QVMJobMetadata, QVMLayer

FAKE_BACKEND = FakeMumbaiV2()

BENCHNAME = "results/vqe_mumbai"


def benchmark_QVM_layer(qasms: list[str], provider: AccountProvider) -> None:
    """Benchmark a QVM layer."""

    RESULT_FILE_PATH = os.path.join(os.getcwd(), f"{BENCHNAME}.csv")

    field_names = ["num_qubits", "fidelity"]
    if not os.path.exists(RESULT_FILE_PATH):
        os.makedirs(os.path.dirname(RESULT_FILE_PATH), exist_ok=True)
        with open(RESULT_FILE_PATH, "w") as csv_file:
            csv.DictWriter(csv_file, fieldnames=field_names).writeheader()

    backend = provider.get_backend("ibmq_qasm_simulator")

    for qasm in qasms:
        qasm_file = os.path.join(os.path.dirname(__file__), qasm)

        circuit = QuantumCircuit.from_qasm_file(qasm_file)
        t_circuit = transpile(circuit, backend=FAKE_BACKEND, optimization_level=3)

        results = backend.run(
            t_circuit, shots=30000, noise_model=NoiseModel.from_backend(FAKE_BACKEND)
        ).result()

        fid = calc_fidelity(circuit, results.get_counts(), provider)

        with open(RESULT_FILE_PATH, "a") as csv_file:
            csv.DictWriter(csv_file, fieldnames=field_names).writerow(
                {
                    field_names[0]: circuit.num_qubits,
                    field_names[1]: fid,
                }
            )


def main():
    qasms = [f"vqe/{i}.qasm" for i in [4, 6, 8, 10, 12, 14, 16, 18, 20]] * 4
    provider = IBMQ.load_account()
    benchmark_QVM_layer(qasms, provider)


if __name__ == "__main__":
    main()
