import csv
import os
import sys
from time import perf_counter

from fidelity import calc_fidelity
from qiskit.circuit import QuantumCircuit
from qiskit.providers.ibmq import AccountProvider

from qvm.stack._types import QVMJobMetadata, QVMLayer


def benchmark_QVM_layer(
    qvm_layer: QVMLayer, qasms: list[str], provider: AccountProvider
) -> None:
    """Benchmark a QVM layer."""

    BENCHNAME = sys.argv[1]

    RESULT_FILE_PATH = os.path.join(os.getcwd(), f"{BENCHNAME}.csv")

    field_names = ["num_qubits", "exec_time", "fidelity"]
    if not os.path.exists(RESULT_FILE_PATH):
        os.makedirs(os.path.dirname(RESULT_FILE_PATH), exist_ok=True)
        with open(RESULT_FILE_PATH, "w") as csv_file:
            csv.DictWriter(csv_file, fieldnames=field_names).writeheader()

    for qasm in qasms:
        qasm_file = os.path.join(os.path.dirname(__file__), qasm)

        circuit = QuantumCircuit.from_qasm_file(qasm_file)

        start = perf_counter()
        job_id = qvm_layer.run(circuit, [], metadata=QVMJobMetadata())

        res = qvm_layer.get_results(job_id)[0]
        end = perf_counter()

        fid = -1.0

        if circuit.num_qubits <= 20:
            counts = res.to_counts(100000)
            fid = calc_fidelity(circuit, counts, provider)

        with open(RESULT_FILE_PATH, "a") as csv_file:
            csv.DictWriter(csv_file, fieldnames=field_names).writerow(
                {
                    field_names[0]: circuit.num_qubits,
                    field_names[1]: end - start,
                    field_names[2]: fid,
                }
            )
