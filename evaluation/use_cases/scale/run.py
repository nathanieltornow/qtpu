from __future__ import annotations

import tracemalloc
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
from qiskit.primitives import BitArray, DataBin, PrimitiveResult, SamplerPubResult
from qiskit.quantum_info import PauliList, SparsePauliOp
from qiskit_addon_cutting import (
    cut_wires,
    expand_observables,
    generate_cutting_experiments,
    partition_problem,
    reconstruct_expectation_values,
)
from qiskit_addon_cutting.automated_cut_finding import (
    DeviceConstraints,
    OptimizationParameters,
    find_cuts,
)

import benchkit as bk

from mqt.bench import get_benchmark_indep

import qtpu
from evaluation.analysis import estimate_runtime

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


def cut_qac(
    circuit: QuantumCircuit, max_qubits: int
) -> tuple[QuantumCircuit, PauliList]:
    circuit = circuit.remove_final_measurements(inplace=False)
    cut_circuit, _ = find_cuts(
        circuit,
        OptimizationParameters(),
        DeviceConstraints(qubits_per_subcircuit=max_qubits),
    )
    qc_w_ancilla = cut_wires(cut_circuit)
    observable = SparsePauliOp(["Z" * circuit.num_qubits])
    observables_expanded = expand_observables(observable.paulis, circuit, qc_w_ancilla)

    return qc_w_ancilla, observables_expanded


def _create_fake_results(circuits: list[QuantumCircuit], shots: int):
    fake_results = []
    for i, circuit in enumerate(circuits):
        data = {}
        for creg in circuit.cregs:
            nbits = creg.size
            counts = BitArray.from_counts(
                {"0" * nbits: shots // 2, "1" * nbits: shots // 2}
            )
            data[creg.name] = counts
        databin = DataBin(**data)
        fake_results.append(
            SamplerPubResult(
                data=databin,
                metadata={"shots": shots},
            )
        )
    return PrimitiveResult(fake_results)


# @bk.catch_failures({"failed": True})
@bk.timeout(3600, {"timeout": True})
def run_qac(
    circuit: QuantumCircuit,
    max_qubits: int,
    num_samples: int,
) -> dict:

    start = perf_counter()
    circuit, observables = cut_qac(circuit, max_qubits=max_qubits)
    compile_time = perf_counter() - start

    partitioned_problem = partition_problem(circuit=circuit, observables=observables)
    subcircuits = partitioned_problem.subcircuits
    subobservables = partitioned_problem.subobservables

    tracemalloc.start()
    start = perf_counter()
    subexperiments, coefficients = generate_cutting_experiments(
        circuits=subcircuits, observables=subobservables, num_samples=num_samples
    )
    generation_time = perf_counter() - start

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    results = {
        label: _create_fake_results(experiment, 1000)
        for label, experiment in subexperiments.items()
    }
    quantum_time = estimate_runtime(
        circuits=[exp for exps in subexperiments.values() for exp in exps],
    )

    start = perf_counter()
    _reconstructed_expval_terms = reconstruct_expectation_values(
        results,
        coefficients,
        subobservables,
    )
    classical_time = perf_counter() - start
    return {
        "compile_time": compile_time,
        "generation_time": generation_time,
        "generation_memory": peak,
        "quantum_time": quantum_time,
        "classical_time": classical_time,
        "num_subcircuits": len(subcircuits),
        "num_experiments": sum(len(exp) for exp in subexperiments.values()),
    }


def run_qtpu(circuit: QuantumCircuit, max_size: int) -> dict[str, float]:
    start = perf_counter()
    circuit = qtpu.cut(circuit, max_size=max_size, cost_weight=1000)
    compile_time = perf_counter() - start

    tracemalloc.start()
    start = perf_counter()
    htn = qtpu.circuit_to_heinsum(circuit)
    qtpu_gen_time = perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    all_circuits = []
    for qt in htn.quantum_tensors:
        all_circuits += qt.flat()

    quantum_time = estimate_runtime(circuits=all_circuits)

    tree, arrays = htn.to_dummy_tn()
    if tree is None:
        classical_time = 0.0

    else:
        start = perf_counter()
        tree.contract(arrays)
        classical_time = perf_counter() - start

    return {
        "compile_time": compile_time,
        "generation_time": qtpu_gen_time,
        "generation_memory": peak,
        "quantum_time": quantum_time,
        "classical_time": classical_time,
        "num_subcircuits": len(htn.quantum_tensors),
        "num_experiments": len(all_circuits),
    }


CIRCUIT_SIZES = list(range(10, 90, 10))
BENCHMARKS = ["qnn"]
SUBCIRC_SIZES = [10]


@bk.foreach(circuit_size=CIRCUIT_SIZES)
@bk.foreach(subcirc_size=SUBCIRC_SIZES)
@bk.foreach(bench=BENCHMARKS)
@bk.log("logs/scale/qtpu.jsonl")
def scale_qtpu_bench(
    circuit_size: int, subcirc_size: int, bench: str
) -> dict[str, float]:
    circuit = get_benchmark_indep(bench, circuit_size=circuit_size, opt_level=3)
    qtpu_metrics = run_qtpu(circuit, max_size=subcirc_size)
    return qtpu_metrics


@bk.foreach(circuit_size=CIRCUIT_SIZES)
@bk.foreach(subcirc_size=SUBCIRC_SIZES)
@bk.foreach(bench=BENCHMARKS)
@bk.log("logs/scale/qac.jsonl")
def scale_qac_bench(
    circuit_size: int, subcirc_size: int, bench: str
) -> dict[str, float]:
    circuit = get_benchmark_indep(bench, circuit_size=circuit_size, opt_level=3)
    qac_metrics = run_qac(
        circuit,
        max_qubits=subcirc_size,
        num_samples=np.inf,
    )
    return qac_metrics


if __name__ == "__main__":
    import sys

    if sys.argv[1] == "qtpu":
        scale_qtpu_bench()
    elif sys.argv[1] == "qac":
        scale_qac_bench()
