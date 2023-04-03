import sys
from time import perf_counter

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.compiler import transpile
from qiskit.providers.fake_provider import FakeMontrealV2
from qiskit.providers.ibmq import IBMQ, AccountProvider
from qiskit.quantum_info import hellinger_fidelity
from qiskit_aer.noise import NoiseModel

from qvm.stack._types import QVMJobMetadata

from stacks import scale_fidelity_stack, scale_time_stack
from util import append_to_csv_file
from fidelity import calc_fidelity


SHOTS = 20000


def two_local_circuit(num_qubits: int, reps: int):
    num_qubits = num_qubits
    circuit = TwoLocal(
        num_qubits=num_qubits,
        rotation_blocks=["ry", "rz", "rx"],
        entanglement="linear",
        entanglement_blocks="rzz",
        reps=reps,
    )
    circuit.measure_all()
    circuit = circuit.decompose()
    params = [
        (np.pi * np.random.uniform(0.0, 1.0)) for _ in range(len(circuit.parameters))
    ]
    return circuit.bind_parameters(params)


def _run_comparison(circuit: QuantumCircuit, provider: AccountProvider) -> float:
    backend = provider.get_backend("ibmq_qasm_simulator")
    fake_backend = FakeMontrealV2()
    t_circuit = transpile(circuit, backend=fake_backend, optimization_level=3)

    results = backend.run(
        t_circuit,
        shots=SHOTS,
        noise_model=NoiseModel.from_backend(fake_backend),
    ).result()

    fid = calc_fidelity(circuit, results.get_counts(), provider)
    return fid


def bench_two_local_fidelity(provider: AccountProvider, reps: int):
    csv_name = f"results/twolocal_comp_fidelity_{reps}-reps.csv"
    stack = scale_fidelity_stack(provider, 4)
    num_qubits = list(range(6, 13, 2)) * 5
    bench_circuits = [two_local_circuit(num_qubits, reps) for num_qubits in num_qubits]
    for bench_circuit in bench_circuits:
        job_id = stack.run(bench_circuit, [], metadata=QVMJobMetadata(shots=SHOTS))
        res = stack.get_results(job_id)[0]
        counts = res.to_counts(SHOTS)
        fid = calc_fidelity(bench_circuit, counts, provider)
        com_fid = _run_comparison(bench_circuit, provider)
        append_to_csv_file(
            csv_name,
            {
                "num_qubits": len(bench_circuit.qubits),
                "qvm_fidelity": fid,
                "comp_fidelity": com_fid,
            },
        )


def bench_two_local_time(sim_size: int, reps: int):
    csv_name = f"results/twolocal_time_{reps}-reps_{sim_size}-qubits.csv"
    stack = scale_time_stack(sim_size)
    num_qubits = list(range(10, 101, sim_size)) * 5
    bench_circuits = [(num_qubits) for num_qubits in num_qubits]
    for bench_circuit in bench_circuits:
        start = perf_counter()
        job_id = stack.run(bench_circuit, [], metadata=QVMJobMetadata(shots=SHOTS))
        res = stack.get_results(job_id)[0]
        runtime = perf_counter() - start
        counts = res.to_counts(SHOTS)
        append_to_csv_file(
            csv_name,
            {
                "num_qubits": len(bench_circuit.qubits),
                "runtime": runtime,
            },
        )


if __name__ == "__main__":
    benchmark = sys.argv[1]
    provider = IBMQ.load_account()
    if benchmark == "fidelity":
        bench_two_local_fidelity(provider, 2)
    elif benchmark == "time":
        bench_two_local_fidelity(int(sys.argv[2]))
    else:
        raise ValueError(f"Unknown benchmark {benchmark}; allowed: fidelity")
