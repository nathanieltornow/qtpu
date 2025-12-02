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
from evaluation.analysis import analyze_heinsum
from evaluation.benchmarks import get_benchmark
from qtpu.compiler.opt._opt import get_pareto_frontier

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit
import math


BENCHMARKS = ["qnn", "wstate", "vqe_su2", "dist-vqe"]
FRACTIONS = [0.25, 0.5, 0.75]
CIRCUIT_SIZES = list(range(20, 141, 20))


def compile_qac(circuit: QuantumCircuit, max_qubits: int) -> dict | None:
    """Run QAC and return a single point with comparable metrics."""
    start = perf_counter()
    cut_circuit, _ = find_cuts(
        circuit,
        OptimizationParameters(),
        DeviceConstraints(qubits_per_subcircuit=max_qubits),
    )
    qc_w_ancilla = cut_wires(cut_circuit)
    heinsum = qtpu.circuit_to_heinsum(qc_w_ancilla)
    compile_time = perf_counter() - start

    analysis = analyze_heinsum(heinsum)

    # Compute comparable metrics
    max_size = max(analysis["qtensor_widths"]) if analysis["qtensor_widths"] else 0
    max_error = max(analysis["qtensor_errors"]) if analysis["qtensor_errors"] else 0

    return {
        "compile_time": compile_time,
        "c_cost": analysis["c_cost"],
        "max_error": max_error,
        "max_size": max_size,
        **analysis,
    }


def compile_qtpu_frontier(
    circuit: QuantumCircuit,
    num_workers: int = 8,
    n_trials: int = 100,
    max_size: int | None = None,
) -> dict:
    """Run QTPU and return the full Pareto frontier."""
    start = perf_counter()
    opt_result = get_pareto_frontier(
        circuit,
        num_workers=num_workers,
        n_trials=n_trials,
    )
    compile_time = perf_counter() - start

    cut_points = opt_result.filter(max_size=max_size)

    # Collect all points with recomputed metrics from HEinsum analysis
    all_points = []
    for p in cut_points:
        cut_circuit = opt_result.get_cut_circuit(p)
        heinsum = qtpu.circuit_to_heinsum(cut_circuit)
        analysis = analyze_heinsum(heinsum)

        max_subcircuit_size = (
            max(analysis["qtensor_widths"]) if analysis["qtensor_widths"] else 0
        )
        max_error = max(analysis["qtensor_errors"]) if analysis["qtensor_errors"] else 0
        c_cost = analysis["c_cost"]

        all_points.append(
            {
                "c_cost": c_cost,
                "max_error": max_error,
                "max_size": max_subcircuit_size,
                "sampling_cost": p.sampling_cost,
            }
        )

    # Recompute Pareto frontier on the actual HEinsum metrics
    # Sort by c_cost, then filter to keep only Pareto-optimal points
    sorted_points = sorted(all_points, key=lambda x: (x["c_cost"], x["max_error"]))
    pareto_points = []
    best_error = float("inf")
    for point in sorted_points:
        if point["max_error"] < best_error:
            pareto_points.append(point)
            best_error = point["max_error"]

    return {
        "compile_time": compile_time,
        "pareto_frontier": pareto_points,
    }


@bk.foreach(bench=BENCHMARKS)
@bk.foreach(circuit_size=CIRCUIT_SIZES)
@bk.foreach(fraction=FRACTIONS)
@bk.log("logs/compiler/qac.jsonl")
def run_qac(bench: str, circuit_size: int, fraction: float) -> dict | None:
    """Run QAC on the specified benchmark and parameters."""
    print(f"Running QAC: bench={bench}, size={circuit_size}, fraction={fraction}")

    # Load benchmark circuit
    circuit = get_benchmark(
        bench, circuit_size=circuit_size, cluster_size=int(circuit_size * fraction)
    ).remove_final_measurements(inplace=False)
    
    # Determine max qubits per subcircuit
    max_qubits = max(2, math.ceil(circuit.num_qubits * fraction))

    # Compile with QAC - returns single point
    results = compile_qac(circuit, max_qubits)
    return results


@bk.foreach(bench=BENCHMARKS)
@bk.foreach(circuit_size=CIRCUIT_SIZES)
@bk.foreach(fraction=FRACTIONS)
@bk.log("logs/compiler/qtpu.jsonl")
def run_qtpu(bench: str, circuit_size: int, fraction: float) -> dict:
    """Run QTPU on the specified benchmark and parameters."""
    print(f"Running QTPU: bench={bench}, size={circuit_size}")

    # Load benchmark circuit
    circuit = get_benchmark(
        bench, circuit_size=circuit_size, cluster_size=10
    ).remove_final_measurements(inplace=False)

    max_size = max(2, math.ceil(circuit.num_qubits * fraction))
    # Compile with QTPU - returns full Pareto frontier
    results = compile_qtpu_frontier(circuit, max_size=max_size)

    return results


if __name__ == "__main__":
    import sys
    import subprocess
    import os

    if len(sys.argv) < 2:
        print("Usage: python run.py [qac|qtpu] [benchmark]")
        print("       python run.py [qac|qtpu] --parallel")
        print("Benchmarks: qnn, wstate, vqe_su2, dist-vqe, or omit for all")
        sys.exit(1)

    cmd = sys.argv[1]
    bench_filter = sys.argv[2] if len(sys.argv) > 2 else None

    if bench_filter == "--parallel":
        # Run all 4 benchmarks in parallel as separate processes
        benchmarks = ["qnn", "wstate", "vqe_su2", "dist-vqe"]
        processes = []
        for bench in benchmarks:
            print(f"Spawning {cmd} for {bench}...")
            p = subprocess.Popen(
                [sys.executable, "-m", "evaluation.compiler.run", cmd, bench],
                cwd=os.getcwd(),
            )
            processes.append((bench, p))
        
        print(f"\nStarted {len(processes)} parallel processes. Waiting for completion...")
        for bench, p in processes:
            p.wait()
            print(f"  {bench}: {'done' if p.returncode == 0 else 'failed'}")
        print("All benchmarks complete.")
    elif cmd == "qac":
        if bench_filter:
            run_qac(bench=bench_filter)
        else:
            run_qac()
    elif cmd == "qtpu":
        if bench_filter:
            run_qtpu(bench=bench_filter)
        else:
            run_qtpu()
    else:
        print("Usage: python run.py [qac|qtpu] [benchmark]")
        print("       python run.py [qac|qtpu] --parallel")
        print("Benchmarks: qnn, wstate, vqe_su2, dist-vqe, or omit for all")
