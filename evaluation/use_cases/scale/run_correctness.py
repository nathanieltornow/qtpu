"""Correctness Validation for Circuit Knitting (Scale Benchmark)

This validates that qTPU's circuit cutting produces correct results by:
1. Cutting a circuit into subcircuits
2. Simulating the subcircuits using Qiskit Aer
3. Reconstructing the expectation value via classical contraction
4. Comparing against ideal full-circuit simulation

This addresses Shepherd Condition 1: end-to-end task-level outcome validation.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_aer import AerSimulator

import benchkit as bk

from mqt.bench import get_benchmark_indep

import cotengra as ctg

import qtpu
from qtpu.transforms import decompose_qpd_measures

if TYPE_CHECKING:
    pass


def compute_ideal_expectation(circuit: QuantumCircuit, observable: str = "Z") -> float:
    """Compute ideal expectation value using statevector simulation.

    Args:
        circuit: The quantum circuit.
        observable: Observable string (default: all-Z).

    Returns:
        Expectation value as float.
    """
    circuit_clean = circuit.remove_final_measurements(inplace=False)
    sv = Statevector.from_instruction(circuit_clean)

    # All-Z observable
    obs_str = observable * circuit_clean.num_qubits
    observable_op = SparsePauliOp([obs_str])

    return float(np.real(sv.expectation_value(observable_op)))


def simulate_circuit_with_aer(circuit: QuantumCircuit, shots: int = 10000) -> float:
    """Simulate circuit with Qiskit Aer and return parity expectation.

    This handles circuits with measurements properly.
    """
    # Add measurements if not present
    if circuit.num_clbits == 0:
        circuit = circuit.copy()
        circuit.measure_all()

    # Run on ideal Aer simulator
    simulator = AerSimulator(method='statevector')
    job = simulator.run(circuit, shots=shots)
    counts = job.result().get_counts()

    # Compute parity expectation (approximates Z-string observable)
    n_even = 0
    n_odd = 0
    for bitstring, count in counts.items():
        parity = bitstring.count("1") % 2
        if parity == 0:
            n_even += count
        else:
            n_odd += count

    return (n_even - n_odd) / shots


def simulate_cut_circuit(circuit: QuantumCircuit, max_size: int, seed: int = 42) -> tuple[float, dict]:
    """Cut and simulate a circuit using qTPU, returning reconstructed expectation value.

    Args:
        circuit: The quantum circuit to cut.
        max_size: Maximum subcircuit size.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (reconstructed expectation value, metrics dict).
    """
    np.random.seed(seed)

    start_total = time.perf_counter()

    # Step 1: Cut the circuit
    start_cut = time.perf_counter()
    cut_circuit = qtpu.cut(circuit, max_size=max_size, cost_weight=1000)
    cut_time = time.perf_counter() - start_cut

    # Step 2: Convert to hEinsum
    start_heinsum = time.perf_counter()
    htn = qtpu.circuit_to_heinsum(cut_circuit)
    heinsum_time = time.perf_counter() - start_heinsum

    # Step 3: Simulate subcircuits using Qiskit Aer
    start_sim = time.perf_counter()

    # Build arrays for ALL tensors: quantum + classical + input
    reconstructed_arrays = []

    # First, handle quantum tensors (need simulation)
    for qt in htn.quantum_tensors:
        # Get all circuit variants from the quantum tensor
        circuits = qt.flat()

        if not circuits:
            # No circuits, use zero array
            reconstructed_arrays.append(np.zeros(qt.shape) if qt.shape else np.array(0.0))
            continue

        # Simulate each circuit variant
        results = []
        for circ in circuits:
            # Decompose ISwitch and QPD operations to standard gates
            circ = circ.decompose()
            circ = decompose_qpd_measures(circ, defer=False, inplace=False)
            circ = circ.decompose()  # Resolve any deferred measurement circuits

            # Use Aer simulator which handles measurements
            expval = simulate_circuit_with_aer(circ)
            results.append(expval)

        # Reshape to tensor shape
        if qt.shape:
            result_array = np.array(results).reshape(qt.shape)
        else:
            result_array = np.array(results[0] if results else 0.0)

        reconstructed_arrays.append(result_array)

    # Add classical tensor data (QPD coefficients)
    for ct in htn.classical_tensors:
        reconstructed_arrays.append(ct.data.numpy())

    sim_time = time.perf_counter() - start_sim

    # Step 4: Classical contraction using optimized contraction tree
    start_contract = time.perf_counter()

    if len(reconstructed_arrays) > 1:
        # Build optimized contraction tree from cotengra
        opt = ctg.HyperOptimizer(parallel=False, progbar=False)
        inputs, outputs = ctg.utils.eq_to_inputs_output(htn.einsum_expr)
        tree = opt.search(inputs, outputs, htn.size_dict)
        reconstructed_value = tree.contract(reconstructed_arrays)
    else:
        # Simple case: just return the array
        reconstructed_value = reconstructed_arrays[0] if reconstructed_arrays else 0.0

    contract_time = time.perf_counter() - start_contract

    total_time = time.perf_counter() - start_total

    # Convert to scalar
    if hasattr(reconstructed_value, 'item'):
        reconstructed_value = float(reconstructed_value.item())
    else:
        reconstructed_value = float(np.mean(reconstructed_value))

    metrics = {
        "cut_time": cut_time,
        "heinsum_time": heinsum_time,
        "sim_time": sim_time,
        "contract_time": contract_time,
        "total_time": total_time,
        "num_subcircuits": len(htn.quantum_tensors),
        "num_circuits": sum(len(qt.flat()) for qt in htn.quantum_tensors),
    }

    return reconstructed_value, metrics


# =============================================================================
# Benchmark Configuration
# =============================================================================

# Small circuits for exact validation
CIRCUIT_SIZES = [10, 15, 20]
SUBCIRCUIT_SIZES = [5, 7, 10]
SEEDS = [0, 1, 2]
# wstate has ideal <Z^n> = -1.0 at all sizes (genuinely entangled, non-trivial to reconstruct)
# qnn has small but nonzero expectation (shows general-case accuracy)
BENCHMARKS = ["wstate", "qnn"]


@bk.foreach(circuit_size=CIRCUIT_SIZES)
@bk.foreach(subcirc_size=SUBCIRCUIT_SIZES)
@bk.foreach(bench=BENCHMARKS)
@bk.foreach(seed=SEEDS)
@bk.log("logs/scale/correctness.jsonl")
@bk.timeout(300, {"timeout": True})
def run_correctness_validation(
    circuit_size: int, subcirc_size: int, bench: str, seed: int
) -> dict:
    """Run correctness validation for circuit knitting."""
    print(f"Correctness: size={circuit_size}, subcirc={subcirc_size}, seed={seed}")

    # Skip if subcircuit is larger than circuit
    if subcirc_size >= circuit_size:
        return {"skipped": True, "reason": "subcirc_size >= circuit_size"}

    # Get benchmark circuit
    circuit = get_benchmark_indep(bench, circuit_size=circuit_size, opt_level=3)

    # Compute ideal expectation value
    try:
        ideal_expval = compute_ideal_expectation(circuit)
    except Exception as e:
        return {"error": str(e), "stage": "ideal_computation"}

    # Cut and simulate
    try:
        reconstructed_expval, metrics = simulate_cut_circuit(
            circuit, max_size=subcirc_size, seed=seed
        )
    except Exception as e:
        return {"error": str(e), "stage": "cut_simulation"}

    # Compute error
    error = abs(reconstructed_expval - ideal_expval)
    relative_error = error / max(abs(ideal_expval), 1e-10)

    return {
        "ideal_expval": ideal_expval,
        "reconstructed_expval": reconstructed_expval,
        "absolute_error": error,
        "relative_error": relative_error,
        **metrics,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        # Quick test
        circuit = get_benchmark_indep("qnn", circuit_size=10, opt_level=3)
        ideal = compute_ideal_expectation(circuit)
        print(f"Ideal expectation: {ideal}")

        reconstructed, metrics = simulate_cut_circuit(circuit, max_size=5)
        print(f"Reconstructed: {reconstructed}")
        print(f"Error: {abs(reconstructed - ideal)}")
        print(f"Metrics: {metrics}")
    else:
        run_correctness_validation()
