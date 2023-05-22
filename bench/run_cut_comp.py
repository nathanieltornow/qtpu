import os
from time import perf_counter

from qiskit.circuit import QuantumCircuit

from qvm.virtual_gates import VirtualBinaryGate, VirtualSWAP
from _util import append_to_csv_file, overhead, find_cut, load_config
from _circuits import get_circuits


def _cut_circuit_bench(
    csv_name: str,
    circuit: QuantumCircuit,
    qpu_size: int = 5,
) -> None:
    fields = [
        "num_qubits",
        "gate_overhead",
        "wire_overhead",
        "optimal_overhead",
        "gate_cut_time",
        "wire_cut_time",
        "optimal_cut_time",
        "gate_num_fragments",
        "wire_num_fragments",
        "optimal_num_fragments",
    ]
    now = perf_counter()
    wire_cut_circ = find_cut(circuit, 10, 0, qpu_size)
    wire_cut_time = perf_counter() - now
    now = perf_counter()
    gate_cut_circ = find_cut(circuit, 0, 10, qpu_size)
    gate_cut_time = perf_counter() - now
    now = perf_counter()
    optimal_cut_circ = find_cut(circuit, 10, 10, qpu_size)
    optimal_cut_time = perf_counter() - now

    append_to_csv_file(
        csv_name,
        {
            fields[0]: circuit.num_qubits,
            fields[1]: overhead(gate_cut_circ),
            fields[2]: overhead(wire_cut_circ),
            fields[3]: overhead(optimal_cut_circ),
            fields[4]: gate_cut_time,
            fields[5]: wire_cut_time,
            fields[6]: optimal_cut_time,
            fields[7]: len(gate_cut_circ.qregs),
            fields[8]: len(wire_cut_circ.qregs),
            fields[9]: len(optimal_cut_circ.qregs),
        },
    )


def run_cut_comp(config: dict):
    qpu_size = config["qpu_size"]

    results_dir = f"results/cut_comp/{qpu_size}"
    os.makedirs(results_dir, exist_ok=True)

    for experiment in config["experiments"]:
        csv_name = f"{results_dir}/{experiment['name']}_{experiment['param']}.csv"
        # run experiment
        nums_qubits = config["nums_qubits"]
        nums_qubits = sorted(nums_qubits * 3)
        circuits = get_circuits(experiment["name"], experiment["param"], nums_qubits)
        for circuit in circuits:
            _cut_circuit_bench(
                csv_name,
                circuit,
                qpu_size,
            )


if __name__ == "__main__":
    run_cut_comp(load_config())
