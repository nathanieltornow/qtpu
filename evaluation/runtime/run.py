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

import benchkit as bk
import qtpu
from qtpu.runtime import HEinsumRuntime, FakeQPUBackend
from qtpu.heinsum import HEinsum
from qtpu.optimize import optimize, OptimizationParameters

from evaluation.benchmarks import get_benchmark


# =============================================================================
# Configuration
# =============================================================================

# Standard MQT benchmarks (linear connectivity, easy for classical)
STANDARD_BENCHMARKS = ["qnn", "wstate", "vqe_su2"]
STANDARD_SIZES = [20, 50, 100, 150]

# Distributed VQE benchmark (all-to-all within clusters, sparse inter-cluster)
DIST_VQE_SIZES = [100]

# Cluster sizes to evaluate (qubits per QPU)
CLUSTER_SIZES = [19]


# =============================================================================
# cuTensorNet TN Simulation Utilities
# =============================================================================


def compute_expval_cutensornet(circuit: QuantumCircuit) -> tuple[float | None, float]:
    """Compute <Z⊗Z⊗...⊗Z> using cuTensorNet contraction.

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

    # Build Pauli string for <Z⊗Z⊗...⊗Z>
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
) -> dict | None:
    """Compile circuit with QTPU and measure execution time.

    Args:
        circuit: Input quantum circuit.
        max_subcircuit_size: Maximum qubits per subcircuit (matches QPU size).

    Returns:
        Dict with timing breakdown and circuit statistics.
    """
    # Compile
    compile_start = perf_counter()
    try:
        # Create HEinsum from circuit and optimize
        heinsum = HEinsum.from_circuit(circuit)
        opt_result = optimize(
            heinsum,
            params=OptimizationParameters(num_workers=8, n_trials=50),
        )

        # Select best point (balances cost and error)
        cut_point = opt_result.select_best(
            max_size=max_subcircuit_size, cost_weight=1000
        )
        if cut_point is None:
            print(f"  No valid cut point selected")
            return None

        # Get optimized HEinsum
        heinsum = opt_result.get_heinsum(cut_point)
    except Exception as e:
        print(f"  Compile failed: {e}")
        return None
    compile_time = perf_counter() - compile_start

    # Create runtime with FakeQPU backend
    backend = FakeQPUBackend(shots=1000)
    runtime = HEinsumRuntime(heinsum, backend=backend, device="cpu")
    runtime.prepare(optimize=True)

    # Execute
    _, timing = runtime.execute()

    return {
        # Timing breakdown
        "compile_time": compile_time,
        "quantum_time": timing.quantum_estimated_qpu_time,
        "classical_contraction_time": timing.classical_contraction_time,
        # Circuit statistics
        "num_subcircuits": timing.num_circuits,
        "num_quantum_tensors": len(heinsum.quantum_tensors),
        "num_classical_tensors": len(heinsum.classical_tensors),
        "contraction_cost": runtime.contraction_cost,
        # Cut point metrics
        "cut_c_cost": cut_point.c_cost,
        "cut_max_error": cut_point.max_error,
        "cut_max_size": cut_point.max_size,
    }


# =============================================================================
# Standard Benchmarks (MQT Bench)
# =============================================================================


@bk.foreach(bench=STANDARD_BENCHMARKS)
@bk.foreach(circuit_size=STANDARD_SIZES)
@bk.foreach(cluster_size=[15])
@bk.log("logs/runtime/standard_qtpu.jsonl")
def run_standard_qtpu(bench: str, circuit_size: int, cluster_size: int) -> dict | None:
    """Run QTPU on standard MQT benchmarks."""
    # Skip if cluster_size > circuit_size
    if cluster_size > circuit_size:
        return None

    print(f"QTPU [{bench}]: size={circuit_size}, cluster={cluster_size}")

    circuit = get_benchmark(bench, circuit_size).remove_final_measurements(
        inplace=False
    )

    result = compile_and_run_qtpu(
        circuit,
        max_subcircuit_size=cluster_size,
    )

    if result is None:
        return None

    return {
        **result,
    }


@bk.foreach(bench=STANDARD_BENCHMARKS)
@bk.foreach(circuit_size=STANDARD_SIZES)
@bk.log("logs/runtime/standard_classical.jsonl")
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


@bk.foreach(circuit_size=DIST_VQE_SIZES)
@bk.foreach(cluster_size=CLUSTER_SIZES)
@bk.log("logs/runtime/dist_vqe_qtpu.jsonl")
def run_dist_vqe_qtpu(circuit_size: int, cluster_size: int) -> dict | None:
    """Run QTPU on distributed VQE circuit."""
    # Skip invalid combinations (cluster must evenly divide circuit)
    if 2 * cluster_size > circuit_size:
        return None

    print(f"QTPU [dist-vqe]: size={circuit_size}, cluster={cluster_size}")

    circuit = get_benchmark("dist-vqe", circuit_size, cluster_size=cluster_size)

    result = compile_and_run_qtpu(
        circuit,
        max_subcircuit_size=cluster_size,
    )

    if result is None:
        return None

    return {
        "circuit_depth": circuit.depth(),
        **result,
    }


@bk.foreach(circuit_size=DIST_VQE_SIZES)
@bk.foreach(cluster_size=CLUSTER_SIZES)
@bk.log("logs/runtime/dist_vqe_classical.jsonl")
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

    if cmd == "standard-qtpu":
        run_standard_qtpu()
    elif cmd == "standard-classical":
        run_standard_classical()
    elif cmd == "standard":
        run_standard_classical()
        run_standard_qtpu()
    elif cmd == "dist-qtpu":
        run_dist_vqe_qtpu()
    elif cmd == "dist-classical":
        run_dist_vqe_classical()
    elif cmd == "dist":
        run_dist_vqe_classical()
        run_dist_vqe_qtpu()
    elif cmd == "all":
        print("Running all evaluations...")
        run_standard_classical()
        run_standard_qtpu()
        run_dist_vqe_classical()
        run_dist_vqe_qtpu()
    else:
        print(f"Unknown command: {cmd}")
        print(usage)
        sys.exit(1)
