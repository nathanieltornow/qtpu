from time import perf_counter
import json
import os
import sys
from multiprocessing import Pool

from qiskit.circuit import QuantumCircuit
from qiskit.providers import Backend
from qiskit.compiler import transpile

import qvm
from qvm.util import virtualize_between_qubits

from _backends import get_backend
from _circuits import get_circuits
from _util import (
    append_to_csv_file,
    calculate_total_variation_distance,
    num_cnots,
    overhead,
    load_config,
)


def _run_vqr_on_circuit(
    circuit: QuantumCircuit,
    backend: Backend,
    csv_name: str,
    num_shots: int = 20000,
    optimization_level: int = 3,
):
    fields = [
        "num_qubits",
        "base_fidelity",
        "vroute_fidelity",
        "base_num_cx",
        "vroute_num_cx",
        "overhead",
        "cut_time",
        "knit_time",
        "run_time",
    ]

    if circuit.num_qubits > backend.num_qubits:
        raise ValueError("Circuit has more qubits than backend.")

    now = perf_counter()
    v_circuit, _ = virtualize_between_qubits(
        circuit, qubit1=circuit.qubits[0], qubit2=circuit.qubits[-1]
    )
    cut_time = perf_counter() - now
    virtualizer = qvm.OneFragmentGateVirtualizer(v_circuit)
    frag, frag_circuit = list(virtualizer.fragments().items())[0]
    args = virtualizer.instantiate()[frag]
    frag_circuit = transpile(
        frag_circuit, backend=backend, optimization_level=optimization_level
    )

    circuits_to_run = [qvm.insert_placeholders(frag_circuit, arg) for arg in args]
    now = perf_counter()
    print("running...")
    counts = backend.run(circuits_to_run, shots=num_shots).result().get_counts()
    assert isinstance(counts, list)
    distrs = [qvm.QuasiDistr.from_counts(count, shots=num_shots) for count in counts]
    run_time = perf_counter() - now
    print("knitting...")
    with Pool() as pool:
        now = perf_counter()
        res_distr = virtualizer.knit({frag: distrs}, pool=pool)
        knit_time = perf_counter() - now

    print("calculating fidelity...")
    t_circuit = transpile(
        circuit, backend=backend, optimization_level=optimization_level
    )
    base_counts = backend.run(t_circuit, shots=num_shots).result().get_counts()
    base_distr = qvm.QuasiDistr.from_counts(base_counts, shots=num_shots)

    from qvm.quasi_distr import QuasiDistr

    fid = calculate_total_variation_distance(
        circuit, QuasiDistr.from_counts(res_distr.to_counts(num_shots))
    )
    base_fid = calculate_total_variation_distance(circuit, base_distr)

    append_to_csv_file(
        csv_name,
        {
            fields[0]: circuit.num_qubits,
            fields[1]: base_fid,
            fields[2]: fid,
            fields[3]: num_cnots(t_circuit),
            fields[4]: num_cnots(frag_circuit),
            fields[5]: overhead(v_circuit),
            fields[6]: cut_time,
            fields[7]: knit_time,
            fields[8]: run_time,
        },
    )


def run_vqr_benchmark(config: dict):
    backend = get_backend(config["backend"])

    # create directory for results
    results_dir = f"results/vqr/{config['backend']}"
    os.makedirs(results_dir, exist_ok=True)

    for experiment in config["experiments"]:
        csv_name = f"{results_dir}/{experiment['name']}_{experiment['param']}.csv"
        # run experiment
        nums_qubits = config["nums_qubits"]
        nums_qubits = sorted(nums_qubits * 3)
        circuits = get_circuits(experiment["name"], experiment["param"], nums_qubits)
        for circuit in circuits:
            _run_vqr_on_circuit(
                circuit,
                backend,
                csv_name,
            )


if __name__ == "__main__":
    run_vqr_benchmark(load_config())
