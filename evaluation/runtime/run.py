"""Runtime evaluation for QTPU on quantum computing workloads.

Compares QTPU (circuit cutting + parallel QPU execution) against classical
tensor network simulation for:
1. Standard benchmarks (MQT Bench: qnn, wstate, vqe_su2)
2. Distributed VQE circuits designed for multi-QPU execution (Khait et al., 2023)

The distributed VQE benchmark is based on "Variational Quantum Eigensolvers in
the Era of Distributed Quantum Computers" (arXiv:2302.14067).
"""

from time import perf_counter

import numpy as np
from qiskit.circuit import QuantumCircuit

import qtpu
from qtpu.core import HEinsum
from qtpu.compiler.opt.optimize import optimize, OptimizationParameters

from evaluation.benchmarks import get_benchmark
from evaluation.utils import log_result


# =============================================================================
# Configuration
# =============================================================================

# Standard MQT benchmarks (linear connectivity, easy for classical)
STANDARD_BENCHMARKS = ["qnn", "wstate", "vqe_su2"]
STANDARD_SIZES = [40, 60, 80, 100]

# Distributed VQE benchmark (all-to-all within clusters, sparse inter-cluster)
DIST_VQE_SIZES = [100]

# Cluster sizes to evaluate (qubits per QPU) — paper Fig 11(c)
CLUSTER_SIZES = list(range(10, 20))

SEEDS = [42, 43, 44]


# =============================================================================
# cuTensorNet TN Simulation Utilities
# =============================================================================


def compute_expval_cutensornet(circuit: QuantumCircuit) -> tuple[float | None, float]:
    """Compute <Z^n> using cuTensorNet contraction.

    Uses NVIDIA cuQuantum's Network API for circuit-to-tensor-network
    conversion and GPU-accelerated contraction with autotuning.

    Args:
        circuit: Qiskit QuantumCircuit to evaluate.

    Returns:
        (expval, time) or (None, time) if failed.
    """
    try:
        import cupy as cp
        from cuquantum import Network, CircuitToEinsum, NetworkOptions
    except ImportError as e:
        print(f"  cuTensorNet not available: {e}")
        return None, 0.0

    n = circuit.num_qubits

    # Build Pauli string for <Z^n>
    pauli_string = "Z" * n

    start = perf_counter()
    try:
        # Convert circuit to einsum using cuQuantum
        converter = CircuitToEinsum(circuit, dtype="complex128", backend=cp)

        # Get expectation value expression (with lightcone optimization)
        expression, operands = converter.expectation(pauli_string, lightcone=True)

        # Use Network API with memory limit to enable automatic slicing
        # memory_limit tells cuTensorNet how much GPU memory is available
        # The slicer will automatically slice indices to fit within this limit
        network_opts = NetworkOptions()  # A40 has 48GB

        with Network(expression, *operands, options=network_opts) as tn:
            # Find contraction path - cuTensorNet will automatically slice
            # based on the memory_limit in NetworkOptions
            path, info = tn.contract_path()
            tn.autotune(iterations=5)

            # Contract
            result = tn.contract()

        # Convert to CPU and get real part
        expval = float(result.real.get()) if hasattr(result, "get") else float(result.real)

        elapsed = perf_counter() - start
        return expval, elapsed
    except Exception as e:
        elapsed = perf_counter() - start
        print(f"  cuTensorNet contraction failed: {e}")
        return None, elapsed


# =============================================================================
# QTPU Evaluation
# =============================================================================


def compile_and_run_qtpu(
    circuit: QuantumCircuit,
    max_subcircuit_size: int,
    seed: int = 42,
) -> dict | None:
    """Compile circuit with QTPU and measure execution time.

    Args:
        circuit: Input quantum circuit.
        max_subcircuit_size: Maximum qubits per subcircuit (matches QPU size).

    Returns:
        Dict with timing breakdown and circuit statistics.
    """
    from qtpu.runtime.baseline import run_heinsum

    # Compile
    compile_start = perf_counter()
    try:
        # Create HEinsum from circuit and optimize
        heinsum = HEinsum.from_circuit(circuit)
        opt_result = optimize(
            heinsum,
            params=OptimizationParameters(num_workers=8, n_trials=150, seed=seed),
        )

        # Select best HEinsum (balances cost and error)
        heinsum = opt_result.select_best(
            max_size=max_subcircuit_size, cost_weight=1000
        )
        if heinsum is None:
            print(f"  No valid cut point selected")
            return None

    except Exception as e:
        print(f"  Compile failed: {e}")
        return None
    compile_time = perf_counter() - compile_start

    # Execute using the baseline run_heinsum function
    _, timing = run_heinsum(heinsum, skip_execution=True)

    return {
        # Timing breakdown
        "compile_time": compile_time,
        "quantum_time": timing.quantum_estimated_qpu_time,
        "classical_contraction_time": timing.classical_contraction_time,
        # Circuit statistics
        "num_subcircuits": timing.num_circuits,
        "num_quantum_tensors": len(heinsum.quantum_tensors),
        "num_classical_tensors": len(heinsum.classical_tensors),
    }


# =============================================================================
# Standard Benchmarks (MQT Bench)
# =============================================================================


def run_standard_qtpu(bench: str, circuit_size: int, cluster_size: int, seed: int = 42) -> dict | None:
    """Run QTPU on standard MQT benchmarks."""
    # Skip if cluster_size > circuit_size
    if cluster_size > circuit_size:
        return None

    print(f"QTPU [{bench}]: size={circuit_size}, cluster={cluster_size}, seed={seed}")

    circuit = get_benchmark(bench, circuit_size).remove_final_measurements(
        inplace=False
    )

    result = compile_and_run_qtpu(
        circuit,
        max_subcircuit_size=cluster_size,
        seed=seed,
    )

    if result is None:
        return None

    return {
        **result,
    }


def run_standard_classical(bench: str, circuit_size: int) -> dict | None:
    """Run classical TN simulation on standard MQT benchmarks using cuTensorNet."""
    print(f"Classical [{bench}]: size={circuit_size}")

    circuit = get_benchmark(bench, circuit_size).remove_final_measurements(
        inplace=False
    )

    # Contract using cuTensorNet
    expval, contract_time = compute_expval_cutensornet(circuit)

    return {
        "contract_time": contract_time,
        "total_time": contract_time,
        "expval": expval,
        "status": "success" if expval is not None else "failed",
    }


# =============================================================================
# Distributed VQE Benchmark (Khait et al., 2023)
# =============================================================================


def run_dist_vqe_qtpu(circuit_size: int, cluster_size: int, seed: int = 42) -> dict | None:
    """Run QTPU on distributed VQE circuit."""
    # Skip invalid combinations (cluster must evenly divide circuit)
    if 2 * cluster_size > circuit_size:
        return None

    print(f"QTPU [dist-vqe]: size={circuit_size}, cluster={cluster_size}, seed={seed}")

    circuit = get_benchmark("dist-vqe", circuit_size, cluster_size=cluster_size)

    result = compile_and_run_qtpu(
        circuit,
        max_subcircuit_size=cluster_size,
        seed=seed,
    )

    if result is None:
        return None

    return {
        "circuit_depth": circuit.depth(),
        **result,
    }


def run_dist_vqe_classical(circuit_size: int, cluster_size: int) -> dict | None:
    """Run classical TN simulation on distributed VQE circuit using cuTensorNet."""
    # Skip invalid combinations (cluster must evenly divide circuit)
    if cluster_size * 2 > circuit_size:
        return None

    print(f"Classical [dist-vqe]: size={circuit_size}, cluster={cluster_size}")

    circuit = get_benchmark("dist-vqe", circuit_size, cluster_size=cluster_size)

    # Contract using cuTensorNet
    expval, contract_time = compute_expval_cutensornet(circuit)

    return {
        "bench": "dist-vqe",
        "circuit_size": circuit_size,
        "cluster_size": cluster_size,
        "circuit_depth": circuit.depth(),
        "contract_time": contract_time,
        "total_time": contract_time,
        "expval": expval,
        "status": "success" if expval is not None else "failed",
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import sys

    usage = """
Runtime Evaluation for QTPU

Usage: python run.py <command>

Commands:
    standard-qtpu       Run QTPU on standard MQT benchmarks
    standard-classical  Run classical TN on standard MQT benchmarks
    standard            Run both QTPU and classical on standard benchmarks

    dist-qtpu           Run QTPU on distributed VQE benchmark
    dist-classical      Run classical TN on distributed VQE benchmark
    dist                Run both QTPU and classical on distributed VQE

    all                 Run all evaluations

Benchmarks:
    Standard (MQT Bench): qnn, wstate, vqe_su2
        - Linear/nearest-neighbor connectivity
        - Easy for classical simulation (low treewidth)

    Distributed VQE (Khait et al., 2023):
        - Dense all-to-all entanglement within clusters
        - Sparse inter-cluster connections (2 CX per boundary)
        - Designed for multi-QPU execution
"""

    if len(sys.argv) < 2:
        print(usage)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd in ("standard-qtpu", "standard", "all"):
        for bench in STANDARD_BENCHMARKS:
            for circuit_size in STANDARD_SIZES:
                for seed in SEEDS:
                    config = {"bench": bench, "circuit_size": circuit_size, "cluster_size": 15, "seed": seed}
                    result = run_standard_qtpu(bench, circuit_size, 15, seed=seed)
                    log_result("logs/runtime/standard_qtpu.jsonl", config, result)

    if cmd in ("standard-classical", "standard", "all"):
        for bench in STANDARD_BENCHMARKS:
            for circuit_size in STANDARD_SIZES:
                config = {"bench": bench, "circuit_size": circuit_size}
                result = run_standard_classical(bench, circuit_size)
                log_result("logs/runtime/standard_classical.jsonl", config, result)

    if cmd in ("dist-qtpu", "dist", "all"):
        for circuit_size in DIST_VQE_SIZES:
            for cluster_size in CLUSTER_SIZES:
                for seed in SEEDS:
                    config = {"circuit_size": circuit_size, "cluster_size": cluster_size, "seed": seed}
                    result = run_dist_vqe_qtpu(circuit_size, cluster_size, seed=seed)
                    log_result("logs/runtime/dist_vqe_qtpu.jsonl", config, result)

    if cmd in ("dist-classical", "dist", "all"):
        for circuit_size in DIST_VQE_SIZES:
            for cluster_size in CLUSTER_SIZES:
                config = {"circuit_size": circuit_size, "cluster_size": cluster_size}
                result = run_dist_vqe_classical(circuit_size, cluster_size)
                log_result("logs/runtime/dist_vqe_classical.jsonl", config, result)

    if cmd not in ("standard-qtpu", "standard-classical", "standard",
                    "dist-qtpu", "dist-classical", "dist", "all"):
        print(f"Unknown command: {cmd}")
        print(usage)
        sys.exit(1)
