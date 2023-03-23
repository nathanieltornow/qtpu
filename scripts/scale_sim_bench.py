import os
import json
from dataclasses import asdict
from datetime import datetime

from qiskit.circuit import QuantumCircuit
from qiskit.providers.ibmq import IBMQ

from qvm.stack._types import QVMJobMetadata
from qvm.stack.qpu_runner import QPURunner
from qvm.stack.decomposer import Decomposer
from qvm.stack.qpus.simulator import IBMQSimulator, LocalSimulator
from qvm.bench import fidelity

# data and time as a string
now_str = datetime.now().strftime("%m-%d-%H-%M-%S")


path = ["hamiltonian", "1_layer"]

nums_of_qubits = [10]

SHOTS = 100000
OUTPUT_FILE = f"scale_sim_{'_'.join(path)}_{now_str}.json"
QASM_FOLDER = "/".join(["benchmark"] + path)


def main():
    provider = IBMQ.load_account()

    qpu = IBMQSimulator(provider)
    qpu_runner = QPURunner(qpus={"sim": qpu})
    stack = Decomposer(qpu_runner, )

    qasm_files = os.listdir(QASM_FOLDER)

    from multiprocessing.pool import Pool

    full_stats = {}

    for n_qubits in nums_of_qubits:
        if f"{n_qubits}.qasm" not in qasm_files:
            print("Skipping", n_qubits)
            continue
        path = f"{QASM_FOLDER}/{n_qubits}.qasm"
        circuit = QuantumCircuit.from_qasm_file(path)
        job_id = stack.run(circuit, [], QVMJobMetadata(qpu_name="sim", shots=100000))
        with Pool() as pool:
            quasi_distr = stack.get_results(job_id, pool)[0]

        counts = quasi_distr.to_counts(SHOTS)
        print(fidelity(circuit, counts, provider))

        stats = stack._stats[job_id]

        stats_dict = asdict(stats)
        stats_json = json.dumps(stats_dict, indent=4)
        print(f"For {n_qubits} qubits:\n")
        print(stats_json)
        full_stats[n_qubits] = stats_dict

    with open(OUTPUT_FILE, "w") as f:
        json.dump(full_stats, f)


if __name__ == "__main__":
    main()
