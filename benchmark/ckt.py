from time import perf_counter

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Sampler
from qiskit.compiler import transpile
from qiskit.primitives import Estimator

from circuit_knitting.cutting import (
    cut_wires,
    expand_observables,
    partition_problem,
    generate_cutting_experiments,
    reconstruct_expectation_values,
)


def run_ckt(
    circuit: QuantumCircuit,
    sampler: Sampler,
    num_samples: int = 1000,
    shots: int = 2**12,
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

    isa_subexperiments = {
        label: [transpile(c) for c in partition_subexpts]
        for label, partition_subexpts in subexperiments.items()
    }

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
        "preparation": preptime,
        "runtime": runtime,
        "postprocessing": posttime,
    }
