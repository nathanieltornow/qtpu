"""
Benchmark: QTPU vs QAC Host Overhead Comparison
===============================================

Compares host-side preparation overhead for circuit knitting between:

1. QAC (Qiskit Addon Cutting):
   Preparation = cutting + partitioning + experiment generation
   
2. QTPU (Quantum Tensor Processing Unit):
   Preparation = cutting + heinsum conversion + optimization + compilation

The key insight: QAC generates exponentially many circuits upfront, while QTPU
uses tensor network contraction to avoid this explosion.
"""

from __future__ import annotations

import tracemalloc
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
from qiskit.primitives import BitArray, DataBin, PrimitiveResult, SamplerPubResult
from qiskit.quantum_info import SparsePauliOp
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
from qtpu.runtime import HEinsumRuntime
from evaluation.analysis import estimate_runtime

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


def _create_fake_results(circuits: list["QuantumCircuit"], shots: int):
    """Create fake sampling results for QAC reconstruction."""
    fake_results = []
    for circuit in circuits:
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


@bk.timeout(3600, {"timeout": True})
def run_qac_overhead(
    circuit: "QuantumCircuit",
    max_qubits: int,
    num_samples: int = np.inf,
) -> dict:
    """Run QAC with detailed timing breakdown of host overhead."""

    # === PREPARATION PHASE ===

    # Step 1: Cutting - find cuts and apply them
    cutting_start = perf_counter()
    circuit_clean = circuit.remove_final_measurements(inplace=False)
    cut_circuit, _ = find_cuts(
        circuit_clean,
        OptimizationParameters(),
        DeviceConstraints(qubits_per_subcircuit=max_qubits),
    )
    qc_w_ancilla = cut_wires(cut_circuit)
    observable = SparsePauliOp(["Z" * circuit_clean.num_qubits])
    observables_expanded = expand_observables(
        observable.paulis, circuit_clean, qc_w_ancilla
    )
    cutting_time = perf_counter() - cutting_start

    # Step 2: Partition problem
    partition_start = perf_counter()
    partitioned_problem = partition_problem(
        circuit=qc_w_ancilla, observables=observables_expanded
    )
    subcircuits = partitioned_problem.subcircuits
    subobservables = partitioned_problem.subobservables
    partition_time = perf_counter() - partition_start

    # Step 3: Generate cutting experiments (exponential blowup happens here!)
    tracemalloc.start()
    generation_start = perf_counter()
    subexperiments, coefficients = generate_cutting_experiments(
        circuits=subcircuits, observables=subobservables, num_samples=num_samples
    )
    generation_time = perf_counter() - generation_start
    _, generation_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Total preparation time
    preparation_time = cutting_time + partition_time + generation_time

    # === EXECUTION PHASE ===

    # Collect all experiment circuits
    all_experiments = [exp for exps in subexperiments.values() for exp in exps]

    # Estimate quantum execution time
    quantum_time = estimate_runtime(circuits=all_experiments)

    # Classical reconstruction
    results_dict = {
        label: _create_fake_results(experiment, 1000)
        for label, experiment in subexperiments.items()
    }
    classical_start = perf_counter()
    _ = reconstruct_expectation_values(
        results_dict,
        coefficients,
        subobservables,
    )
    classical_time = perf_counter() - classical_start

    return {
        # Preparation breakdown
        "cutting_time": cutting_time,
        "partition_time": partition_time,
        "generation_time": generation_time,
        "preparation_time": preparation_time,
        "generation_memory": generation_memory,
        # Execution
        "quantum_time": quantum_time,
        "classical_time": classical_time,
        # Stats
        "num_subcircuits": len(subcircuits),
        "num_experiments": len(all_experiments),
    }


@bk.timeout(3600, {"timeout": True})
def run_qtpu_overhead(
    circuit: "QuantumCircuit",
    max_size: int,
) -> dict:
    """Run QTPU with detailed timing breakdown of host overhead."""

    # === PREPARATION PHASE ===

    # Step 1: Cutting - find optimal cuts
    cutting_start = perf_counter()
    cut_circuit = qtpu.cut(circuit, max_size=max_size, cost_weight=1000)
    cutting_time = perf_counter() - cutting_start

    # Step 2: Convert to HEinsum
    tracemalloc.start()
    generation_start = perf_counter()
    heinsum = qtpu.circuit_to_heinsum(cut_circuit)
    generation_time = perf_counter() - generation_start
    _, generation_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Step 3: Prepare runtime (optimization + backend compilation)
    backend = FakeQPUCudaQBackend(shots=1000)
    runtime = HEinsumRuntime(heinsum, backend=backend, device="cpu")

    prepare_start = perf_counter()
    runtime.prepare(optimize=True)
    prepare_time = perf_counter() - prepare_start

    # Extract detailed timing from prepare()
    prep_timing = runtime.prep_timing
    optimization_time = prep_timing.optimization_time if prep_timing else 0.0
    compilation_time = prep_timing.circuit_compilation_time if prep_timing else 0.0

    # Total preparation time
    preparation_time = cutting_time + generation_time + prepare_time

    # === EXECUTION PHASE ===

    # Execute and get timing
    _, timing = runtime.execute()

    # Collect circuit stats
    all_circuits = []
    for qt in heinsum.quantum_tensors:
        all_circuits += qt.flat()

    return {
        # Preparation breakdown
        "cutting_time": cutting_time,
        "generation_time": generation_time,
        "optimization_time": optimization_time,
        "compilation_time": compilation_time,
        "prepare_time": prepare_time,
        "preparation_time": preparation_time,
        "generation_memory": generation_memory,
        # Execution
        "quantum_time": timing.quantum_estimated_qpu_time,
        "classical_time": timing.classical_contraction_time,
        # Stats
        "num_subcircuits": len(heinsum.quantum_tensors),
        "num_experiments": len(all_circuits),
    }


# =============================================================================
# BENCHMARK CONFIGURATION
# =============================================================================

CIRCUIT_SIZES = list(range(10, 90, 10))
BENCHMARKS = ["qnn"]
SUBCIRC_SIZES = [10]


@bk.foreach(circuit_size=CIRCUIT_SIZES)
@bk.foreach(subcirc_size=SUBCIRC_SIZES)
@bk.foreach(bench=BENCHMARKS)
@bk.log("logs/scale/qtpu_overhead.jsonl")
def bench_qtpu_overhead(
    circuit_size: int, subcirc_size: int, bench: str
) -> dict[str, float]:
    """Benchmark QTPU host overhead."""
    circuit = get_benchmark_indep(bench, circuit_size=circuit_size, opt_level=3)
    return run_qtpu_overhead(circuit, max_size=subcirc_size)


@bk.foreach(circuit_size=CIRCUIT_SIZES)
@bk.foreach(subcirc_size=SUBCIRC_SIZES)
@bk.foreach(bench=BENCHMARKS)
@bk.log("logs/scale/qac_overhead.jsonl")
def bench_qac_overhead(
    circuit_size: int, subcirc_size: int, bench: str
) -> dict[str, float]:
    """Benchmark QAC host overhead."""
    circuit = get_benchmark_indep(bench, circuit_size=circuit_size, opt_level=3)
    return run_qac_overhead(circuit, max_qubits=subcirc_size)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python run_overhead.py [qtpu|qac|both]")
        sys.exit(1)

    if sys.argv[1] == "qtpu":
        bench_qtpu_overhead()
    elif sys.argv[1] == "qac":
        bench_qac_overhead()
    elif sys.argv[1] == "both":
        bench_qtpu_overhead()
        bench_qac_overhead()
