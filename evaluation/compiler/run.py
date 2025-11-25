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
from qtpu.compiler._opt import get_pareto_frontier
from qtpu.tensor import HybridTensorNetwork

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


def prepend_dict_keys(d: dict, prefix: str) -> dict:
    return {f"{prefix}{k}": v for k, v in d.items()}


def compile_qac(circuit: QuantumCircuit, max_qubits: int) -> dict:
    start = perf_counter()
    cut_circuit, _ = find_cuts(
        circuit,
        OptimizationParameters(),
        DeviceConstraints(qubits_per_subcircuit=max_qubits),
    )
    qc_w_ancilla = cut_wires(cut_circuit)

    htn = qtpu.circuit_to_hybrid_tn(qc_w_ancilla)
    compile_time = perf_counter() - start
    return {"compile_time": compile_time, **analyze_hybrid_tn(htn)}


def compile_qtpu(
    circuit: QuantumCircuit,
    max_sampling_cost: float = 1e6,
    num_workers: int = 8,
    num_trials: int = 50,
) -> dict:
    """Compile a QuantumCircuit into a HybridTensorNetwork representation."""
    start = perf_counter()
    cut_circuit = qtpu.cut(
        circuit,
        max_sampling_cost=max_sampling_cost,
        num_workers=num_workers,
        n_trials=num_trials,
    )

    htn = qtpu.circuit_to_hybrid_tn(cut_circuit)
    compile_time = perf_counter() - start

    return {"compile_time": compile_time, **analyze_hybrid_tn(htn)}


def get_qtpu_best_for_budget(
    circuit: QuantumCircuit,
    target_c_cost: float,
    num_workers: int = 8,
    num_trials: int = 50,
) -> dict:
    """Find QTPU's best solution within a classical cost budget.
    
    Explores the Pareto frontier and returns the solution with lowest max_error
    that has c_cost <= target_c_cost.
    """
    start = perf_counter()
    # Explore with high sampling cost to get many options
    frontier = get_pareto_frontier(
        circuit,
        max_sampling_cost=200,
        num_workers=num_workers, 
        n_trials=num_trials
    )
    compile_time = perf_counter() - start
    
    # Find best solution within budget
    valid_points = [p for p in frontier if p["c_cost"] <= target_c_cost]
    
    if not valid_points:
        # If no solution within budget, return the one with lowest c_cost
        best = min(frontier, key=lambda p: p["c_cost"])
    else:
        # Return the one with lowest max_error within budget
        best = min(valid_points, key=lambda p: p["max_error"])
    
    return {
        "compile_time": compile_time,
        "c_cost": best["c_cost"],
        "max_error": best["max_error"],
        "quantum_cost": best["quantum_cost"],
    }


# ============================================================================
# Benchmark configurations
# ============================================================================
BENCHMARKS = ["wstate"]
SIZES = [50, 75, 100]
# Subcircuit size as fraction of total: 1/2, 1/3, 1/4, 1/5
FRACTIONS = [2, 3, 4, 5]


# ============================================================================
# QAC Benchmark: vary subcircuit fraction (1/2, 1/3, 1/4, 1/5 of circuit size)
# ============================================================================
@bk.foreach(bench=BENCHMARKS)
@bk.foreach(circuit_size=SIZES)
@bk.foreach(fraction=FRACTIONS)
@bk.log("logs/compile/qac.jsonl")
def compile_qac_benchmark(bench: str, circuit_size: int, fraction: int = 2) -> dict:
    circuit = get_benchmark_indep(bench, circuit_size).remove_final_measurements(
        inplace=False
    )
    max_qubits = circuit_size // fraction
    return compile_qac(circuit, max_qubits=max_qubits)


# ============================================================================
# Direct Comparison: QTPU vs QAC at same classical cost budget
# For each QAC config, run QTPU with the same c_cost budget
# ============================================================================
@bk.foreach(bench=BENCHMARKS)
@bk.foreach(circuit_size=SIZES)
@bk.foreach(fraction=FRACTIONS)
@bk.log("logs/compile/comparison.jsonl")
def compile_comparison_benchmark(bench: str, circuit_size: int, fraction: int = 2) -> dict:
    """Compare QTPU vs QAC at the same classical cost budget."""
    circuit = get_benchmark_indep(bench, circuit_size).remove_final_measurements(
        inplace=False
    )
    
    # First run QAC to get its c_cost and max_error
    max_qubits = circuit_size // fraction
    qac_result = compile_qac(circuit, max_qubits=max_qubits)
    qac_c_cost = qac_result["c_cost"]
    qac_max_error = max(qac_result["qtensor_errors"])
    
    # Now run QTPU with the same c_cost budget
    qtpu_result = get_qtpu_best_for_budget(
        circuit, 
        target_c_cost=qac_c_cost,
        num_workers=8,
        num_trials=50
    )
    
    return {
        "qac_c_cost": qac_c_cost,
        "qac_max_error": qac_max_error,
        "qac_compile_time": qac_result["compile_time"],
        "qtpu_c_cost": qtpu_result["c_cost"],
        "qtpu_max_error": qtpu_result["max_error"],
        "qtpu_compile_time": qtpu_result["compile_time"],
        "qtpu_quantum_cost": qtpu_result["quantum_cost"],
        "error_reduction": (qac_max_error - qtpu_result["max_error"]) / qac_max_error * 100,
    }


if __name__ == "__main__":
    import sys

    if "qac" in sys.argv:
        compile_qac_benchmark()
    elif "comparison" in sys.argv:
        compile_comparison_benchmark()
    else:
        print("Usage: python -m evaluation.compiler.run [qac|comparison]")
