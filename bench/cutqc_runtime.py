import os
import csv
from dataclasses import dataclass, asdict
from time import perf_counter

import numpy as np
from qiskit.circuit import QuantumCircuit

import qvm
from qvm import QVMCompiler
from qvm.compiler.virtualization import OptimalDecompositionPass
from qvm.compiler.distr_transpiler.backend_mapper import BasicBackendMapper
from qiskit.providers.fake_provider import FakeKolkataV2

from cutqc import (
    cut_circuit_wires,
    evaluate_subcircuits,
    reconstruct_full_distribution,
)

from circuits.circuits import get_circuits


NUM_THREADS = 8


@dataclass
class BenchmarkResult:
    num_qubits: int
    qpu_size: int
    sim_time: np.nan
    knit_time: np.nan
    sim_time_base: np.nan
    knit_time_base: np.nan

    def append_dict_to_csv(self, filepath: str) -> None:
        data = asdict(self)
        if not os.path.exists(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w") as csv_file:
                csv.DictWriter(csv_file, fieldnames=data.keys()).writeheader()
                csv.DictWriter(csv_file, fieldnames=data.keys()).writerow(data)
            return

        with open(filepath, "a") as csv_file:
            csv.DictWriter(csv_file, fieldnames=data.keys()).writerow(data)


def cut_hamsim(circuit: QuantumCircuit, size_to_reach: int):
    num_binary_gates = sum(1 for instr in circuit if len(instr.qubits) == 2)

    num_cuts = int(np.ceil((num_binary_gates + 1) / size_to_reach))

    subcircuit_vertices = [
        list(range(size_to_reach * i, size_to_reach * (i + 1)))
        for i in range(num_cuts - 1)
    ]
    subcircuit_vertices.append(
        list(range(size_to_reach * (num_cuts - 1), num_binary_gates))
    )

    # return cut_circuit_wires(
    #     circuit=circuit,
    #     method="automatic",
    #     max_subcircuit_width=size_to_reach,
    #     max_cuts=3,
    #     num_subcircuits=[2, 3, 4],
    # )

    return cut_circuit_wires(
        circuit, method="manual", subcircuit_vertices=subcircuit_vertices, verbose=False
    )


def time_benchmark(circuit_size: int, size_to_reach: int) -> BenchmarkResult:
    circuit = get_circuits("hamsim_1", (circuit_size, circuit_size + 1))[0]

    # CutQC benchmark
    _remove_mesurements(circuit)
    cuts = cut_hamsim(circuit, size_to_reach)

    now = perf_counter()

    results = evaluate_subcircuits(cuts)
    run_time_base = perf_counter() - now

    now = perf_counter()
    _ = reconstruct_full_distribution(circuit, results, cuts, num_threads=NUM_THREADS)
    knit_time_base = perf_counter() - now

    print("CutQC benchmark:", run_time_base, knit_time_base)

    # QVM benchmark

    circuit.measure_all()
    comp = QVMCompiler(
        [OptimalDecompositionPass(size_to_reach=size_to_reach)],
        [BasicBackendMapper(FakeKolkataV2())],
    )
    vc = comp.run(circuit, budget=3)
    _, times = qvm.run(vc, optimization_level=3, num_processes=NUM_THREADS)

    run_time = times.qpu_time
    knit_time = times.knit_time

    print("QVM benchmark:", run_time, knit_time)
    return BenchmarkResult(
        circuit.num_qubits,
        size_to_reach,
        run_time,
        knit_time,
        run_time_base,
        knit_time_base,
    )


def _remove_mesurements(circuit: QuantumCircuit) -> None:
    circuit.remove_final_measurements(inplace=True)
    cregs = circuit.cregs.copy()
    for creg in cregs:
        circuit.remove_register(creg)


if __name__ == "__main__":
    workload = [(20, 8), (20, 12), (20, 10)]
    for circuit_size, qpu_size in workload:
        result = time_benchmark(circuit_size, qpu_size)
        result.append_dict_to_csv(f"bench/results/cutqc_runtime/{circuit_size}.csv")
