import tracemalloc
from time import perf_counter

import numpy as np
from mqt.bench import get_benchmark_indep
from qiskit.circuit import QuantumCircuit
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
from qiskit_aer.primitives import SamplerV2
from qiskit_ibm_runtime.fake_provider import FakeMontrealV2

from evaluation.analysis import estimate_runtime

import benchkit as bk
import qtpu


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


def run_qac(
    circuit: QuantumCircuit,
    observables: PauliList,
    num_samples: int,
    shots: int,
) -> tuple[float, dict]:

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
    reconstructed_expval_terms = reconstruct_expectation_values(
        results,
        coefficients,
        subobservables,
    )
    classical_time = perf_counter() - start
    return reconstructed_expval_terms[0], {
        "qac.generation_time": generation_time,
        "qac.generation_memory": peak,
        "qac.quantum_time": quantum_time,
        "qac.classical_time": classical_time,
        "qac.num_subcircuits": len(subcircuits),
        "qac.num_experiments": sum(len(exp) for exp in subexperiments.values()),
    }


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


def run_qtpu(circuit: QuantumCircuit) -> dict[str, float]:
    tracemalloc.start()
    start = perf_counter()
    htn = qtpu.circuit_to_hybrid_tn(circuit)
    qtpu_gen_time = perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    all_circuits = []
    for qt in htn.qtensors:
        all_circuits += qt.flat()

    quantum_time = estimate_runtime(
        circuits=all_circuits,
        backend=FakeMontrealV2(),
        shots=1000,
    )

    tn = htn.to_dummy_tn()
    start = perf_counter()
    tn.contract(all, optimize="auto-hq", output_inds=[])
    classical_time = perf_counter() - start

    return {
        "qtpu.generation_time": qtpu_gen_time,
        "qtpu.generation_memory": peak,
        "qtpu.quantum_time": quantum_time,
        "qtpu.classical_time": classical_time,
        "qtpu.num_subcircuits": len(htn.subcircuits),
        "qtpu.num_experiments": len(all_circuits),
    }


@bk.foreach(circuit_size=[10, 20, 30, 40])
@bk.foreach(subcirc_size=[10])
@bk.foreach(bench=["qnn", "wstate"])
@bk.foreach(num_samples=[100000])
@bk.foreach(_repeat=list(range(5)))
@bk.log("logs/01_scale.jsonl")
def scale_bench(
    bench: str, circuit_size: int, subcirc_size: int, num_samples: int, _repeat: int
) -> QuantumCircuit:
    circuit = get_benchmark_indep(bench, circuit_size=circuit_size, opt_level=3)

    start = perf_counter()
    cut_circ, observables = cut_circuit(circuit, max_qubits=subcirc_size)
    cut_time = perf_counter() - start

    _, qac_metrics = run_qac(
        cut_circ,
        observables,
        num_samples=num_samples,
        shots=1000,
    )

    qtpu_metrics = run_qtpu(cut_circ)

    return {
        "qac.cut_time": cut_time,
        **qac_metrics,
        **qtpu_metrics,
    }
