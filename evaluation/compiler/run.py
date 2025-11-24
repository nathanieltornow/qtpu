import os
from time import perf_counter
from typing import TYPE_CHECKING

from mqt.bench import get_benchmark_indep
from qiskit.circuit import QuantumCircuit
from qiskit_addon_cutting import (
    cut_wires,
)
from qiskit_addon_cutting.automated_cut_finding import (
    DeviceConstraints,
    OptimizationParameters,
    find_cuts,
)

import benchkit as bk
import qtpu
from evaluation.analysis import analyze_hybrid_tn
from qtpu.tensor import HybridTensorNetwork

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


def prepend_dict_keys(d: dict, prefix: str) -> dict:
    return {f"{prefix}{k}": v for k, v in d.items()}


def compile_qac(circuit: QuantumCircuit, max_qubits: int) -> HybridTensorNetwork:
    start = perf_counter()
    cut_circuit, _ = find_cuts(
        circuit,
        OptimizationParameters(),
        DeviceConstraints(qubits_per_subcircuit=max_qubits),
    )
    qc_w_ancilla = cut_wires(cut_circuit)

    htn = qtpu.circuit_to_hybrid_tn(qc_w_ancilla)
    compile_time = perf_counter() - start
    return prepend_dict_keys(
        {"compile_time": compile_time, **analyze_hybrid_tn(htn)}, "qac."
    )


def compile_qtpu(
    circuit: QuantumCircuit,
    gamma_q: float,
    gamma_c: float,
    num_threads: int = 1,
    num_trials: int = 1,
) -> HybridTensorNetwork:
    """Compile a QuantumCircuit into a HybridTensorNetwork representation.

    Parameters:
        circuit (QuantumCircuit): The quantum circuit to be compiled.

    Returns:
        HybridTensorNetwork: The compiled hybrid tensor network representation of the circuit.
    """
    # --- Placeholder for compilation logic ---
    start = perf_counter()
    cut_circuit = qtpu.cut(
        circuit,
        gamma_q=gamma_q,
        gamma_c=gamma_c,
        num_threads=num_threads,
        n_trials=num_trials,
    )

    htn = qtpu.circuit_to_hybrid_tn(cut_circuit)
    compile_time = perf_counter() - start

    return prepend_dict_keys(
        {"compile_time": compile_time, **analyze_hybrid_tn(htn)}, "qtpu."
    )


BENCHMARKS = ["qnn", "graphstate", "wstate", "vqe_su2"]
SIZES = list(range(10, 101, 10))


@bk.foreach(bench=BENCHMARKS)
@bk.foreach(circuit_size=SIZES)
@bk.foreach(gamma_q=[1.05, 1.1, 1.2], gamma_c=[500.0, 1000.0, 2000.0])
@bk.foreach(num_trials=[10, 50, 100], num_threads=[1, 1, 1])
@bk.log("logs/compile/qtpu.jsonl")
def compile_qtpu_benchmark(
    bench: str,
    circuit_size: int,
    gamma_q: float,
    gamma_c: float,
    num_threads: int = 1,
    num_trials: int = 1,
) -> dict:
    circuit = get_benchmark_indep(bench, circuit_size).remove_final_measurements(
        inplace=False
    )
    return compile_qtpu(
        circuit,
        gamma_q=gamma_q,
        gamma_c=gamma_c,
        num_threads=num_threads,
        num_trials=num_trials,
    )


@bk.foreach(bench=BENCHMARKS)
@bk.foreach(circuit_size=SIZES)
@bk.foreach(max_qubits=[10, 20, 30])
@bk.log("logs/compile/qac.jsonl")
def compile_qac_benchmark(bench: str, circuit_size: int, max_qubits: int = 20) -> dict:
    circuit = get_benchmark_indep(bench, circuit_size).remove_final_measurements(
        inplace=False
    )
    return compile_qac(circuit, max_qubits=max_qubits)


if __name__ == "__main__":

    import sys

    if "qtpu" in sys.argv:
        compile_qtpu_benchmark()
    else:
        compile_qac_benchmark()
