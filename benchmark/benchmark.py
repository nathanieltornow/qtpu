import csv
import os
import sys
from datetime import datetime
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

    now_str = datetime.now().strftime("%m-%d-%H-%M-%S")
    dirname = os.path.dirname(__file__)

    RESULT_FILE = os.path.join("bench_results", f"{BENCHNAME}_{now_str}.csv")

    field_names = ["num_qubits", "exec_time", "fidelity"]

    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
    with open(RESULT_FILE, "w") as csv_file:
        csv.DictWriter(csv_file, fieldnames=field_names).writeheader()

    for qasm in qasms:
        qasm_file = os.path.join(dirname, qasm)

        circuit = QuantumCircuit.from_qasm_file(qasm_file)

        start = perf_counter()
        job_id = qvm_layer.run(circuit, [], metadata=QVMJobMetadata())

        res = qvm_layer.get_results(job_id)[0]
        end = perf_counter()

        fid = -1.0

        if circuit.num_qubits <= 20:
            counts = res.to_counts(100000)
            fid = calc_fidelity(circuit, counts, provider)

        with open(RESULT_FILE, "a") as csv_file:
            csv.DictWriter(csv_file, fieldnames=field_names).writerow(
                {
                    field_names[0]: circuit.num_qubits,
                    field_names[1]: end - start,
                    field_names[2]: fid,
                }
            )
