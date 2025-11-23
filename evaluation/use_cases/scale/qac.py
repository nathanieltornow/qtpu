from __future__ import annotations

import tracemalloc
from time import perf_counter
from typing import TYPE_CHECKING

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
from qiskit_ibm_runtime.fake_provider import FakeMontrealV2

import benchkit as bk
from evaluation.analysis import estimate_runtime

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


def cut_circuit(
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


@bk.timeout(600, {"timeout": True})
def run_qac(
    circuit: QuantumCircuit,
    observables: PauliList,
    num_samples: int,
    shots: int,
) -> dict:

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
        label: _create_fake_results(experiment, shots)
        for label, experiment in subexperiments.items()
    }
    quantum_time = estimate_runtime(
        circuits=[exp for exps in subexperiments.values() for exp in exps],
        backend=FakeMontrealV2(),
        shots=shots,
    )

    start = perf_counter()
    _reconstructed_expval_terms = reconstruct_expectation_values(
        results,
        coefficients,
        subobservables,
    )
    classical_time = perf_counter() - start
    return {
        "qac.generation_time": generation_time,
        "qac.generation_memory": peak,
        "qac.quantum_time": quantum_time,
        "qac.classical_time": classical_time,
        "qac.num_subcircuits": len(subcircuits),
        "qac.num_experiments": sum(len(exp) for exp in subexperiments.values()),
    }
