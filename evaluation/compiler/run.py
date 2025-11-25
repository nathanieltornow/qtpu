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


def analyze_htn(htn: HybridTensorNetwork) -> dict:
    """Analyze a HybridTensorNetwork and return metrics."""
    max_error = max(
        sum(0.01 if inst.operation.num_qubits == 2 else 0.001 for inst in sc.data)
        for sc in htn.subcircuits
    )
    c_cost = htn.to_dummy_tn().contraction_cost(optimize="auto")
    max_width = max(sc.num_qubits for sc in htn.subcircuits)
    return {
        "c_cost": c_cost,
        "max_error": max_error,
        "max_width": max_width,
        "num_subcircuits": len(htn.subcircuits),
    }


def compile_qac(circuit: QuantumCircuit, max_qubits: int) -> dict | None:
    """Run QAC and return results, or None if it fails."""
    try:
        start = perf_counter()
        cut_circuit, _ = find_cuts(
            circuit,
            OptimizationParameters(),
            DeviceConstraints(qubits_per_subcircuit=max_qubits),
        )
        qc_w_ancilla = cut_wires(cut_circuit)
        htn = qtpu.circuit_to_hybrid_tn(qc_w_ancilla)
        compile_time = perf_counter() - start
        
        return {"compile_time": compile_time, **analyze_htn(htn)}
    except Exception as e:
        print(f"QAC failed: {e}")
        return None


def compile_qtpu_auto(circuit: QuantumCircuit, num_workers: int = 8, n_trials: int = 50) -> dict:
    """Run QTPU with automatic best_tradeoff selection."""
    start = perf_counter()
    cut_circuit = qtpu.cut(circuit, num_workers=num_workers, n_trials=n_trials)
    htn = qtpu.circuit_to_hybrid_tn(cut_circuit)
    compile_time = perf_counter() - start
    
    return {"compile_time": compile_time, **analyze_htn(htn)}


def compute_efficiency(baseline_error: float, c_cost: float, max_error: float) -> float:
    """Compute efficiency: error reduction per unit classical cost.
    
    Higher is better. Returns error_reduction / log(1 + c_cost) to handle
    the large range of c_cost values.
    """
    import math
    error_reduction = baseline_error - max_error
    if c_cost == 0:
        return float('inf') if error_reduction > 0 else 0
    return error_reduction / math.log(1 + c_cost)


# ============================================================================
# Benchmark configurations
# ============================================================================
BENCHMARKS = ["wstate", "ghz"]
SIZES = [20, 30]  # Start small since QAC is very slow
FRACTIONS = [2, 3, 4]  # 1/2, 1/3, 1/4 of circuit size


# ============================================================================
# Comprehensive Comparison: QTPU Pareto frontier vs QAC discrete points
# The RIGHT comparison: at each c_cost level, who gets lower error?
# ============================================================================
@bk.foreach(bench=BENCHMARKS)
@bk.foreach(circuit_size=SIZES)
@bk.log("logs/compile/comparison.jsonl")
def compile_comparison_benchmark(bench: str, circuit_size: int) -> dict:
    """Compare QTPU's Pareto frontier vs all QAC configurations.
    
    The key insight: QTPU provides a continuous Pareto frontier, while QAC
    only provides discrete points (1/2, 1/3, 1/4, etc.). The fair comparison is:
    - At each QAC c_cost level, what error does QTPU achieve?
    - Does QTPU provide useful intermediate points that QAC cannot hit?
    """
    from time import perf_counter
    circuit = get_benchmark_indep(bench, circuit_size).remove_final_measurements(
        inplace=False
    )

    # Compute baseline (no cuts)
    baseline_error = sum(
        0.01 if inst.operation.num_qubits == 2 else 0.001 for inst in circuit.data
    )

    # Get QTPU's full Pareto frontier
    start = perf_counter()
    frontier = get_pareto_frontier(circuit, max_sampling_cost=200, n_trials=50, num_workers=8)
    qtpu_compile_time = perf_counter() - start
    
    # Sort frontier by c_cost
    frontier = sorted(frontier, key=lambda p: p["c_cost"])

    # Run QAC at all fractions
    qac_results = {}
    for fraction in FRACTIONS:
        max_qubits = circuit_size // fraction
        result = compile_qac(circuit, max_qubits=max_qubits)
        if result is not None:
            qac_results[f"1/{fraction}"] = result

    # Compare at each QAC c_cost level: what error can QTPU achieve?
    qtpu_at_qac_costs = {}
    for qac_key, qac in qac_results.items():
        qac_c_cost = qac["c_cost"]
        # Find QTPU point with closest c_cost <= qac_c_cost
        valid_qtpu = [p for p in frontier if p["c_cost"] <= qac_c_cost]
        if valid_qtpu:
            best_qtpu = min(valid_qtpu, key=lambda p: p["max_error"])
            qtpu_at_qac_costs[qac_key] = {
                "qtpu_c_cost": best_qtpu["c_cost"],
                "qtpu_error": best_qtpu["max_error"],
                "qac_c_cost": qac_c_cost,
                "qac_error": qac["max_error"],
                "qtpu_better": best_qtpu["max_error"] <= qac["max_error"],
                "error_improvement": qac["max_error"] - best_qtpu["max_error"],
            }
    
    # Count wins, ties, losses
    wins = sum(1 for v in qtpu_at_qac_costs.values() if v["qtpu_error"] < v["qac_error"])
    ties = sum(1 for v in qtpu_at_qac_costs.values() if abs(v["qtpu_error"] - v["qac_error"]) < 0.001)
    losses = sum(1 for v in qtpu_at_qac_costs.values() if v["qtpu_error"] > v["qac_error"] + 0.001)
    
    # Count QTPU-only points (c_costs that QAC cannot hit)
    qac_c_costs = set(qac["c_cost"] for qac in qac_results.values())
    qtpu_only_points = [p for p in frontier if p["c_cost"] not in qac_c_costs and p["c_cost"] > 0]

    return {
        # Baseline
        "baseline_error": baseline_error,
        
        # QTPU Pareto frontier
        "qtpu_frontier_size": len(frontier),
        "qtpu_compile_time": qtpu_compile_time,
        "qtpu_frontier": [{"c_cost": p["c_cost"], "max_error": p["max_error"], "quantum_cost": p["quantum_cost"]} for p in frontier],
        
        # QAC discrete points
        "qac_points": {k: {"c_cost": v["c_cost"], "max_error": v["max_error"], "compile_time": v["compile_time"]} for k, v in qac_results.items()},
        
        # Head-to-head at each QAC c_cost level
        "comparison_at_qac_costs": qtpu_at_qac_costs,
        
        # Summary
        "qtpu_wins": wins,
        "qtpu_ties": ties,
        "qtpu_losses": losses,
        "qtpu_only_points": len(qtpu_only_points),
        
        # Verdict: QTPU wins if it matches/beats QAC at all levels AND provides extra points
        "qtpu_dominates": (losses == 0) and (len(qtpu_only_points) > 0),
    }


# ============================================================================
# Scalability Benchmark: How fast can each compiler reach target subcircuit sizes?
# ============================================================================
SCALABILITY_BENCHMARKS = ["wstate", "ghz", "qnn"]
SCALABILITY_SIZES = [20, 30, 40, 50, 60, 70, 80]
TARGET_FRACTIONS = [2, 3, 4, 5]  # Target 1/2, 1/3, 1/4, 1/5 of original size


@bk.foreach(bench=SCALABILITY_BENCHMARKS)
@bk.foreach(circuit_size=SCALABILITY_SIZES)
@bk.foreach(target_fraction=TARGET_FRACTIONS)
@bk.log("logs/compile/scalability.jsonl")
def compile_scalability_benchmark(bench: str, circuit_size: int, target_fraction: int) -> dict:
    """Measure compile time to achieve target subcircuit size for QTPU vs QAC.
    
    For each (benchmark, size, fraction):
    - Run QTPU to get a cut with max_width <= circuit_size / fraction
    - Run QAC with the same constraint
    - Compare compile times and achieved metrics
    """
    from time import perf_counter
    from qtpu.compiler._opt import cut_at_target
    
    target_width = circuit_size // target_fraction
    
    circuit = get_benchmark_indep(bench, circuit_size).remove_final_measurements(
        inplace=False
    )
    
    # Compute baseline
    baseline_error = sum(
        0.01 if inst.operation.num_qubits == 2 else 0.001 for inst in circuit.data
    )
    
    result = {
        "target_width": target_width,
        "target_fraction": f"1/{target_fraction}",
        "baseline_error": baseline_error,
    }
    
    # Run QTPU with max_subcircuit_width constraint
    try:
        start = perf_counter()
        cut_circuit = cut_at_target(
            circuit, 
            target_quantum_cost=target_width,
            num_workers=8, 
            n_trials=50
        )
        qtpu_time = perf_counter() - start
        
        htn = qtpu.circuit_to_hybrid_tn(cut_circuit)
        qtpu_metrics = analyze_htn(htn)
        
        result["qtpu_compile_time"] = qtpu_time
        result["qtpu_achieved_width"] = qtpu_metrics["max_width"]
        result["qtpu_c_cost"] = qtpu_metrics["c_cost"]
        result["qtpu_max_error"] = qtpu_metrics["max_error"]
        result["qtpu_success"] = qtpu_metrics["max_width"] <= target_width
    except Exception as e:
        result["qtpu_compile_time"] = None
        result["qtpu_error"] = str(e)
        result["qtpu_success"] = False
    
    # Run QAC with same constraint
    try:
        start = perf_counter()
        cut_circuit, _ = find_cuts(
            circuit,
            OptimizationParameters(),
            DeviceConstraints(qubits_per_subcircuit=target_width),
        )
        qc_w_ancilla = cut_wires(cut_circuit)
        htn = qtpu.circuit_to_hybrid_tn(qc_w_ancilla)
        qac_time = perf_counter() - start
        
        qac_metrics = analyze_htn(htn)
        
        result["qac_compile_time"] = qac_time
        result["qac_achieved_width"] = qac_metrics["max_width"]
        result["qac_c_cost"] = qac_metrics["c_cost"]
        result["qac_max_error"] = qac_metrics["max_error"]
        result["qac_success"] = qac_metrics["max_width"] <= target_width
    except Exception as e:
        result["qac_compile_time"] = None
        result["qac_error"] = str(e)
        result["qac_success"] = False
    
    # Compute speedup if both succeeded
    if result.get("qtpu_compile_time") and result.get("qac_compile_time"):
        result["speedup"] = result["qac_compile_time"] / result["qtpu_compile_time"]
    
    return result


if __name__ == "__main__":
    import sys

    if "comparison" in sys.argv:
        compile_comparison_benchmark()
    elif "scalability" in sys.argv:
        compile_scalability_benchmark()
    else:
        print("Usage: python -m evaluation.compiler.run [comparison|scalability]")
