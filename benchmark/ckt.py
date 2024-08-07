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
) -> float:
    observable = SparsePauliOp(["Z" * circuit.num_qubits])

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

    isa_subexperiments = {
        label: [transpile(c) for c in partition_subexpts]
        for label, partition_subexpts in subexperiments.items()
    }
    results = {
        label: sampler.run(subsystem_subexpts, shots=shots).result()
        for label, subsystem_subexpts in subexperiments.items()
    }
    reconstructed_expvals = reconstruct_expectation_values(
        results,
        coefficients,
        subobservables,
    )

    return reconstructed_expvals[0]
