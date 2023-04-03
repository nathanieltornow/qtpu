import sys
from time import perf_counter

from qiskit.circuit import QuantumCircuit
from qiskit.providers.ibmq import IBMQ, AccountProvider
from qiskit.quantum_info import hellinger_fidelity

from qvm.stack._types import QVMJobMetadata

from stacks import scale_fidelity_stack, scale_time_stack
from util import append_to_csv_file


SHOTS = 20000


def ghz_circuit(num_qubits: int):
    circuit = QuantumCircuit(num_qubits)
    circuit.h(0)
    circuit.cx(list(range(num_qubits - 1)), list(range(1, num_qubits)))
    circuit.measure_all()
    return circuit


def bench_ghz_fidelity(provider: AccountProvider):
    csv_name = "results/ghz_fidelity.csv"
    stack = scale_fidelity_stack(provider, 4)
    num_qubits = list(range(4, 21, 2)) * 5
    bench_circuits = [ghz_circuit(num_qubits) for num_qubits in num_qubits]
    for bench_circuit in bench_circuits:
        job_id = stack.run(bench_circuit, [], metadata=QVMJobMetadata(shots=SHOTS))
        res = stack.get_results(job_id)[0]
        ghz_state = {
            "0" * bench_circuit.num_qubits: 0.5 * SHOTS,
            "1" * bench_circuit.num_qubits: 0.5 * SHOTS,
        }
        counts = res.to_counts(SHOTS)
        fid = hellinger_fidelity(ghz_state, counts)
        append_to_csv_file(
            csv_name,
            {"num_qubits": len(bench_circuit.qubits), "fidelity": fid},
        )


def bench_ghz_time(sim_size: int):
    csv_name = f"results/ghz_time_{sim_size}-qubits.csv"
    stack = scale_time_stack(sim_size)
    num_qubits = list(range(10, 101, sim_size)) * 5
    bench_circuits = [ghz_circuit(num_qubits) for num_qubits in num_qubits]
    for bench_circuit in bench_circuits:
        start = perf_counter()
        job_id = stack.run(bench_circuit, [], metadata=QVMJobMetadata(shots=SHOTS))
        res = stack.get_results(job_id)[0]
        runtime = perf_counter() - start
        ghz_state = {
            "0" * bench_circuit.num_qubits: 0.5 * SHOTS,
            "1" * bench_circuit.num_qubits: 0.5 * SHOTS,
        }
        counts = res.to_counts(SHOTS)
        fid = hellinger_fidelity(ghz_state, counts)
        append_to_csv_file(
            csv_name,
            {
                "num_qubits": len(bench_circuit.qubits),
                "fidelity": fid,
                "runtime": runtime,
            },
        )


if __name__ == "__main__":
    benchmark = sys.argv[1]
    provider = IBMQ.load_account()
    if benchmark == "fidelity":
        bench_ghz_fidelity(provider)
    elif benchmark == "time":
        bench_ghz_time(int(sys.argv[2]))
    else:
        raise ValueError(f"Unknown benchmark {benchmark}; allowed: fidelity")

