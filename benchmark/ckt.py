from time import perf_counter
from typing import Callable

import numpy as np

from qiskit.primitives import Sampler
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.compiler import transpile

from circuit_knitting.cutting import (
    cut_wires,
    expand_observables,
    partition_problem,
    generate_cutting_experiments,
    reconstruct_expectation_values,
)
from circuit_knitting.cutting.automated_cut_finding import (
    find_cuts,
    OptimizationParameters,
    DeviceConstraints,
)

from qtpu.contract import evaluate_hybrid_tn
from qtpu.evaluate import evaluate_sampler
from qtpu.circuit import circuit_to_hybrid_tn


def run_qtpu(
    circuit: QuantumCircuit,
    tolerance: float = 0.0,
    eval_fn: Callable[[list[QuantumCircuit]], list] | None = None,
) -> tuple[float, dict]:
    start = perf_counter()
    htn = circuit_to_hybrid_tn(circuit)
    htn.simplify(tolerance)
    for qt in htn.quantum_tensors:
        qt.generate_instances()

    preptime = perf_counter() - start

    start = perf_counter()
    tn = evaluate_hybrid_tn(htn, eval_fn)
    runtime = perf_counter() - start

    start = perf_counter()
    res = tn.contract(optimize="auto", output_inds=[])
    posttime = perf_counter() - start

    return res, {
        "qtpu_preparation": preptime,
        "qtpu_runtime": runtime,
        "qtpu_postprocessing": posttime,
    }


def run_ckt(
    circuit: QuantumCircuit,
    sampler: Sampler,
    num_samples: int = np.inf,
    shots: int = 20000,
    obs: str | None = None,
) -> tuple[float, dict]:
    if obs is None:
        obs = "Z" * circuit.num_qubits
    observable = SparsePauliOp([obs])

    start = perf_counter()
    qc_w_ancilla = cut_wires(circuit)
    observables_expanded = expand_observables(observable.paulis, circuit, qc_w_ancilla)
    partitioned_problem = partition_problem(
        circuit=circuit, observables=observables_expanded
    )

    subcircuits = partitioned_problem.subcircuits
    subobservables = partitioned_problem.subobservables
    subexperiments, coefficients = generate_cutting_experiments(
        circuits=subcircuits, observables=subobservables, num_samples=num_samples
    )

    preptime = perf_counter() - start

    # isa_subexperiments = {
    #     label: [transpile(c) for c in partition_subexpts]
    #     for label, partition_subexpts in subexperiments.items()
    # }

    start = perf_counter()
    results = {
        label: sampler.run(subsystem_subexpts, shots=shots).result()
        for label, subsystem_subexpts in subexperiments.items()
    }
    runtime = perf_counter() - start

    start = perf_counter()
    reconstructed_expvals = reconstruct_expectation_values(
        results,
        coefficients,
        subobservables,
    )
    posttime = perf_counter() - start

    return reconstructed_expvals[0], {
        "ckt_preparation": preptime,
        "ckt_runtime": runtime,
        "ckt_postprocessing": posttime,
    }


def cut_ckt(circuit: QuantumCircuit, subcircuit_size: int) -> QuantumCircuit:
    optimization_settings = OptimizationParameters()

    # Specify the size of the QPUs available
    device_constraints = DeviceConstraints(qubits_per_subcircuit=subcircuit_size)

    cut_circuit, metadata = find_cuts(
        circuit, optimization_settings, device_constraints
    )
    print(
        f'Found solution using {len(metadata["cuts"])} cuts with a sampling '
        f'overhead of {metadata["sampling_overhead"]}.'
    )
    return cut_circuit
