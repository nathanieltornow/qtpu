"""Hardware Simulation Benchmark

This validates qTPU on realistic hardware simulation by:
1. Using Qiskit Aer with noise model from real IBM device calibration
2. Comparing estimated vs simulated QPU performance
3. Showing that more classical assistance (more cutting) improves fidelity

This addresses Shepherd Condition 2: real quantum hardware results.
(Using calibrated noise model as proxy for actual hardware execution)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeBrisbane, FakeCusco

import cotengra as ctg

import benchkit as bk

from mqt.bench import get_benchmark_indep

import qtpu
from qtpu.transforms import decompose_qpd_measures
from evaluation.analysis import estimate_runtime

if TYPE_CHECKING:
    pass


# =============================================================================
# Noise Model Setup
# =============================================================================

def get_device_noise_model(device_name: str = "brisbane") -> tuple[NoiseModel, object]:
    """Get noise model from fake IBM device.

    Args:
        device_name: Name of fake device ("brisbane", "marrakesh").

    Returns:
        Tuple of (noise_model, fake_backend).
    """
    if device_name == "brisbane":
        fake_backend = FakeBrisbane()
    elif device_name == "cusco":
        fake_backend = FakeCusco()
    else:
        raise ValueError(f"Unknown device: {device_name}")

    noise_model = NoiseModel.from_backend(fake_backend)
    return noise_model, fake_backend


# =============================================================================
# Ideal Simulation
# =============================================================================

def compute_ideal_expectation(circuit: QuantumCircuit) -> float:
    """Compute ideal expectation value using statevector simulation."""
    circuit_clean = circuit.remove_final_measurements(inplace=False)
    sv = Statevector.from_instruction(circuit_clean)

    # All-Z observable
    observable = SparsePauliOp(["Z" * circuit_clean.num_qubits])
    return float(np.real(sv.expectation_value(observable)))


# =============================================================================
# Noisy Hardware Simulation
# =============================================================================

def run_hardware_simulation(
    circuit: QuantumCircuit,
    noise_model: NoiseModel,
    backend,
    shots: int = 10000,
) -> tuple[float, float, float]:
    """Run noisy hardware simulation.

    Returns:
        Tuple of (expectation value, std error, wall clock time).
    """
    start_time = time.perf_counter()

    # Add measurements if not present
    if circuit.num_clbits == 0:
        circuit = circuit.copy()
        circuit.measure_all()

    # Transpile for backend
    transpiled = transpile(circuit, backend, optimization_level=3)

    # Run on noisy simulator
    simulator = AerSimulator(noise_model=noise_model)
    result = simulator.run(transpiled, shots=shots).result()
    counts = result.get_counts()

    wall_time = time.perf_counter() - start_time

    # Compute expectation value from counts
    n_even = 0
    n_odd = 0
    for bitstring, count in counts.items():
        parity = bitstring.count("1") % 2
        if parity == 0:
            n_even += count
        else:
            n_odd += count

    expval = (n_even - n_odd) / shots
    std_err = np.sqrt((1 - expval**2) / shots)

    return expval, std_err, wall_time


# =============================================================================
# qTPU Circuit Knitting on Hardware Simulation
# =============================================================================

def run_qtpu_hardware(
    circuit: QuantumCircuit,
    max_subcircuit_size: int,
    noise_model: NoiseModel,
    backend,
    shots: int = 10000,
) -> dict:
    """Run qTPU circuit knitting with hardware noise simulation.

    Args:
        circuit: Original circuit to cut.
        max_subcircuit_size: Maximum qubits per subcircuit.
        noise_model: Device noise model.
        backend: Backend for transpilation.
        shots: Number of shots per subcircuit.

    Returns:
        Dict with results and metrics.
    """
    start_total = time.perf_counter()

    # Step 1: Cut the circuit
    start_cut = time.perf_counter()
    cut_circuit = qtpu.cut(circuit, max_size=max_subcircuit_size, cost_weight=1000)
    cut_time = time.perf_counter() - start_cut

    # Step 2: Convert to hEinsum
    start_heinsum = time.perf_counter()
    htn = qtpu.circuit_to_heinsum(cut_circuit)
    heinsum_time = time.perf_counter() - start_heinsum

    # Step 3: Simulate each subcircuit with hardware noise
    start_sim = time.perf_counter()

    subcircuit_results = []
    for qt in htn.quantum_tensors:
        circuits = qt.flat()

        if not circuits:
            subcircuit_results.append([0.0])
            continue

        results = []
        for circ in circuits:
            # Decompose ISwitch and QPD operations to standard gates
            circ = circ.decompose()
            circ = decompose_qpd_measures(circ, defer=False, inplace=False)
            # Decompose again to resolve any deferred measurement circuits
            circ = circ.decompose()

            expval, _, _ = run_hardware_simulation(circ, noise_model, backend, shots=shots)
            results.append(expval)

        subcircuit_results.append(results)

    sim_time = time.perf_counter() - start_sim

    # Step 4: Classical reconstruction using einsum
    start_contract = time.perf_counter()

    # Build arrays for contraction - need all tensors: quantum + classical
    reconstructed_arrays = []
    for qt, results in zip(htn.quantum_tensors, subcircuit_results):
        if qt.shape:
            arr = np.array(results).reshape(qt.shape)
        else:
            arr = np.array(results[0] if results else 0.0)
        reconstructed_arrays.append(arr)

    # Add classical tensor data (QPD coefficients)
    for ct in htn.classical_tensors:
        reconstructed_arrays.append(ct.data.numpy())

    # Contract using optimized contraction tree
    if len(reconstructed_arrays) > 1:
        opt = ctg.HyperOptimizer(parallel=False, progbar=False)
        inputs, outputs = ctg.utils.eq_to_inputs_output(htn.einsum_expr)
        tree = opt.search(inputs, outputs, htn.size_dict)
        reconstructed_value = tree.contract(reconstructed_arrays)
    else:
        reconstructed_value = reconstructed_arrays[0] if reconstructed_arrays else 0.0

    contract_time = time.perf_counter() - start_contract

    total_time = time.perf_counter() - start_total

    # Convert to scalar
    if hasattr(reconstructed_value, 'item'):
        reconstructed_value = float(reconstructed_value.item())
    else:
        reconstructed_value = float(np.mean(reconstructed_value))

    # Estimate QPU time (decompose for accurate estimate)
    all_circuits = []
    for qt in htn.quantum_tensors:
        for circ in qt.flat():
            circ = circ.decompose()
            circ = decompose_qpd_measures(circ, defer=False, inplace=False)
            circ = circ.decompose()
            all_circuits.append(circ)
    estimated_qpu_time = estimate_runtime(circuits=all_circuits) if all_circuits else 0.0

    return {
        "reconstructed_expval": reconstructed_value,
        "cut_time": cut_time,
        "heinsum_time": heinsum_time,
        "sim_time": sim_time,
        "contract_time": contract_time,
        "total_time": total_time,
        "estimated_qpu_time": estimated_qpu_time,
        "num_subcircuits": len(htn.quantum_tensors),
        "num_circuits": len(all_circuits),
        "max_subcircuit_qubits": max(len(circ.qubits) for circ in all_circuits) if all_circuits else 0,
    }


# =============================================================================
# Benchmark Functions
# =============================================================================

CIRCUIT_SIZES = [10, 15, 20]  # Small for noise simulation
SUBCIRCUIT_SIZES = [5, 7, 10]  # Different cutting granularities
DEVICES = ["brisbane"]
# Use wstate: ideal <Z^n> = -1.0 at all sizes, giving meaningful error metrics
BENCH_CIRCUIT = "wstate"


@bk.foreach(circuit_size=CIRCUIT_SIZES)
@bk.foreach(subcirc_size=SUBCIRCUIT_SIZES)
@bk.foreach(device=DEVICES)
@bk.log("logs/hardware/qtpu_noisy.jsonl")
@bk.timeout(600, {"timeout": True})
def bench_qtpu_hardware(circuit_size: int, subcirc_size: int, device: str) -> dict:
    """Benchmark qTPU on hardware noise simulation."""
    print(f"Hardware: size={circuit_size}, subcirc={subcirc_size}, device={device}")

    # Skip if subcircuit is larger than circuit
    if subcirc_size >= circuit_size:
        return {"skipped": True, "reason": "subcirc_size >= circuit_size"}

    # Get circuit and noise model
    circuit = get_benchmark_indep(BENCH_CIRCUIT, circuit_size=circuit_size, opt_level=3)
    noise_model, fake_backend = get_device_noise_model(device)

    # Compute ideal expectation
    try:
        ideal_expval = compute_ideal_expectation(circuit)
    except Exception as e:
        return {"error": str(e), "stage": "ideal"}

    # Run qTPU with hardware noise
    try:
        result = run_qtpu_hardware(
            circuit,
            max_subcircuit_size=subcirc_size,
            noise_model=noise_model,
            backend=fake_backend,
            shots=5000,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "stage": "qtpu"}

    # Compute error metrics
    absolute_error = abs(result["reconstructed_expval"] - ideal_expval)
    relative_error = absolute_error / max(abs(ideal_expval), 1e-10)

    return {
        "ideal_expval": ideal_expval,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        **result,
    }


@bk.foreach(circuit_size=CIRCUIT_SIZES)
@bk.foreach(device=DEVICES)
@bk.log("logs/hardware/direct_noisy.jsonl")
@bk.timeout(600, {"timeout": True})
def bench_direct_hardware(circuit_size: int, device: str) -> dict:
    """Benchmark direct execution on hardware noise simulation (no cutting)."""
    print(f"Direct Hardware: size={circuit_size}, device={device}")

    # Get circuit and noise model
    circuit = get_benchmark_indep(BENCH_CIRCUIT, circuit_size=circuit_size, opt_level=3)
    noise_model, fake_backend = get_device_noise_model(device)

    # Compute ideal expectation
    try:
        ideal_expval = compute_ideal_expectation(circuit)
    except Exception as e:
        return {"error": str(e), "stage": "ideal"}

    # Run direct on noisy simulator
    try:
        noisy_expval, std_err, wall_time = run_hardware_simulation(
            circuit, noise_model, fake_backend, shots=10000
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "stage": "simulation"}

    # Compute error metrics
    absolute_error = abs(noisy_expval - ideal_expval)
    relative_error = absolute_error / max(abs(ideal_expval), 1e-10)

    return {
        "ideal_expval": ideal_expval,
        "noisy_expval": noisy_expval,
        "std_error": std_err,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "wall_time": wall_time,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python run.py [qtpu|direct|all|quick]")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "qtpu":
        bench_qtpu_hardware()
    elif cmd == "direct":
        bench_direct_hardware()
    elif cmd == "all":
        print("Running all hardware benchmarks...")
        bench_qtpu_hardware()
        bench_direct_hardware()
    elif cmd == "quick":
        # Quick test
        circuit = get_benchmark_indep("qnn", circuit_size=10, opt_level=3)
        noise_model, fake_backend = get_device_noise_model("brisbane")

        print("Ideal:", compute_ideal_expectation(circuit))

        noisy, std, _ = run_hardware_simulation(circuit, noise_model, fake_backend)
        print(f"Direct noisy: {noisy} ± {std}")

        result = run_qtpu_hardware(circuit, max_subcircuit_size=5, noise_model=noise_model, backend=fake_backend)
        print(f"qTPU (cut to 5q): {result['reconstructed_expval']}")
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
