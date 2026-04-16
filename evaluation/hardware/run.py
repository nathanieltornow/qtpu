"""Real IBM Quantum Hardware Benchmark (OSDI Condition 2)
=======================================================

Runs circuit benchmarks on IBM Marrakesh and validates the
FakeMarrakesh QPU time estimation model.

Workloads:
  - QNN at 20, 40, 60, 80, 100 qubits (cut to 10-qubit subcircuits)
  - W-State, VQE-SU2, Dist-VQE (if credits/queue permit)

Metrics:
  - Actual QPU wall time (from IBM job metadata)
  - Actual fidelity (Hellinger distance vs ideal simulation)
  - Estimated QPU time (FakeMarrakesh + ASAP scheduling)
  - Estimation error: |actual - estimated| / actual

Fallback:
  If hardware access is limited, use AerSimulator with
  NoiseModel.from_backend(FakeMarrakesh()) for noisy simulation.
"""

from __future__ import annotations

import json
import sys
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2, Batch
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

import benchkit as bk

import qtpu
from qtpu.core import HEinsum
from qtpu.compiler.opt.optimize import optimize, OptimizationParameters

from evaluation.analysis import estimate_runtime
from evaluation.benchmarks import get_benchmark

if TYPE_CHECKING:
    pass


# =============================================================================
# Configuration
# =============================================================================

BACKEND_NAME = "ibm_marrakesh"
QPU_SIZE = 10
SHOTS = 1000

# Primary benchmarks
QNN_SIZES = [20, 40, 60, 80, 100]

# Secondary benchmarks (if credits permit)
SECONDARY_BENCHMARKS = ["wstate", "vqe_su2"]
SECONDARY_SIZES = [20, 40, 60]


# =============================================================================
# Hardware Execution
# =============================================================================


def get_ibm_backend(channel: str = "ibm_quantum"):
    """Connect to IBM Quantum and get the Marrakesh backend."""
    service = QiskitRuntimeService(channel=channel)
    return service.backend(BACKEND_NAME)


def compile_qtpu_subcircuits(
    circuit: QuantumCircuit,
    max_size: int,
) -> tuple[HEinsum, list[QuantumCircuit], float]:
    """Compile circuit with qTPU and extract subcircuits.

    Returns:
        (heinsum, flat_circuits, compile_time)
    """
    start = perf_counter()
    heinsum = HEinsum.from_circuit(circuit)
    opt_result = optimize(
        heinsum,
        params=OptimizationParameters(num_workers=8, n_trials=150),
    )
    heinsum = opt_result.select_best(max_size=max_size, cost_weight=1000)
    compile_time = perf_counter() - start

    if heinsum is None:
        return None, [], compile_time

    # Extract all concrete subcircuits
    flat_circuits = []
    for qt in heinsum.quantum_tensors:
        flat_circuits.extend(qt.flat())

    return heinsum, flat_circuits, compile_time


def transpile_for_hardware(
    circuits: list[QuantumCircuit], backend
) -> list[QuantumCircuit]:
    """Transpile circuits for real hardware execution."""
    from qtpu.transforms import remove_operations_by_name

    clean_circuits = [
        remove_operations_by_name(c, {"qpd_measure", "iswitch"}, inplace=False)
        for c in circuits
    ]
    return transpile(
        circuits=clean_circuits,
        backend=backend,
        optimization_level=3,
        scheduling_method="asap",
    )


def run_on_hardware(
    circuits: list[QuantumCircuit],
    backend,
    shots: int = SHOTS,
) -> tuple[list[dict], float]:
    """Submit circuits to IBM hardware via SamplerV2.

    Returns:
        (results, actual_qpu_time)
    """
    with Batch(backend=backend) as batch:
        sampler = SamplerV2(mode=batch)
        job = sampler.run(circuits, shots=shots)
        result = job.result()

    # Extract actual execution time from job metadata
    metadata = job.metrics()
    actual_qpu_time = metadata.get("usage", {}).get("quantum_seconds", 0.0)

    # Extract measurement results
    measurement_results = []
    for pub_result in result:
        counts = pub_result.data.meas.get_counts()
        measurement_results.append(counts)

    return measurement_results, actual_qpu_time


def compute_hellinger_fidelity(
    counts_real: dict, counts_ideal: dict, shots: int
) -> float:
    """Compute Hellinger fidelity between real and ideal distributions."""
    all_keys = set(counts_real.keys()) | set(counts_ideal.keys())
    fidelity = 0.0
    for key in all_keys:
        p = counts_real.get(key, 0) / shots
        q = counts_ideal.get(key, 0) / shots
        fidelity += np.sqrt(p * q)
    return fidelity**2


def simulate_ideal(circuits: list[QuantumCircuit], shots: int = SHOTS) -> list[dict]:
    """Run noiseless simulation for fidelity comparison."""
    from qiskit_aer import AerSimulator

    sim = AerSimulator()
    # Add measurements if not present
    meas_circuits = []
    for circ in circuits:
        if not any(
            instr.operation.name == "measure" for instr in circ.data
        ):
            c = circ.copy()
            c.measure_all()
            meas_circuits.append(c)
        else:
            meas_circuits.append(circ)

    results = sim.run(meas_circuits, shots=shots).result()
    return [results.get_counts(i) for i in range(len(meas_circuits))]


# =============================================================================
# Noisy Simulation Fallback
# =============================================================================


def run_noisy_simulation(
    circuits: list[QuantumCircuit], shots: int = SHOTS
) -> tuple[list[dict], float]:
    """Run circuits on AerSimulator with IBM Marrakesh noise model.

    Fallback when real hardware credits/queue are limited.
    """
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel

    fake_backend = FakeMarrakesh()
    noise_model = NoiseModel.from_backend(fake_backend)

    sim = AerSimulator(noise_model=noise_model)
    transpiled = transpile(circuits, backend=sim, optimization_level=3)

    result = sim.run(transpiled, shots=shots).result()

    # Estimate QPU time from scheduling (same as FakeMarrakesh model)
    estimated_time = estimate_runtime(circuits)

    counts = [result.get_counts(i) for i in range(len(circuits))]
    return counts, estimated_time


# =============================================================================
# Benchmark Functions
# =============================================================================


@bk.foreach(circuit_size=QNN_SIZES)
@bk.log("logs/hardware/qnn_hardware.jsonl")
def bench_qnn_hardware(circuit_size: int) -> dict | None:
    """Run QNN benchmark on real IBM hardware."""
    print(f"Hardware QNN: {circuit_size} qubits")

    circuit = get_benchmark("qnn", circuit_size).remove_final_measurements(
        inplace=False
    )

    # Compile with qTPU
    heinsum, flat_circuits, compile_time = compile_qtpu_subcircuits(
        circuit, max_size=QPU_SIZE
    )
    if heinsum is None:
        print(f"  No valid partition found")
        return None

    print(f"  Subcircuits: {len(flat_circuits)}")

    # Estimate QPU time (FakeMarrakesh model)
    estimated_qpu_time = estimate_runtime(flat_circuits)

    # Get hardware backend and transpile
    backend = get_ibm_backend()
    transpiled = transpile_for_hardware(flat_circuits, backend)

    # Run on real hardware
    hw_results, actual_qpu_time = run_on_hardware(transpiled, backend)

    # Run ideal simulation for fidelity
    ideal_results = simulate_ideal(flat_circuits)

    # Compute per-subcircuit fidelity
    fidelities = []
    for hw_counts, ideal_counts in zip(hw_results, ideal_results):
        fid = compute_hellinger_fidelity(hw_counts, ideal_counts, SHOTS)
        fidelities.append(fid)

    return {
        "compile_time": compile_time,
        "estimated_qpu_time": estimated_qpu_time,
        "actual_qpu_time": actual_qpu_time,
        "estimation_error": abs(actual_qpu_time - estimated_qpu_time)
        / max(actual_qpu_time, 1e-9),
        "mean_fidelity": float(np.mean(fidelities)),
        "min_fidelity": float(np.min(fidelities)),
        "num_subcircuits": len(flat_circuits),
        "num_quantum_tensors": len(heinsum.quantum_tensors),
    }


@bk.foreach(circuit_size=QNN_SIZES)
@bk.log("logs/hardware/qnn_noisy_sim.jsonl")
def bench_qnn_noisy_sim(circuit_size: int) -> dict | None:
    """Fallback: Run QNN benchmark with noisy simulation."""
    print(f"Noisy Sim QNN: {circuit_size} qubits")

    circuit = get_benchmark("qnn", circuit_size).remove_final_measurements(
        inplace=False
    )

    heinsum, flat_circuits, compile_time = compile_qtpu_subcircuits(
        circuit, max_size=QPU_SIZE
    )
    if heinsum is None:
        return None

    estimated_qpu_time = estimate_runtime(flat_circuits)

    # Run noisy simulation
    noisy_results, sim_time = run_noisy_simulation(flat_circuits)

    # Run ideal simulation
    ideal_results = simulate_ideal(flat_circuits)

    fidelities = []
    for noisy_counts, ideal_counts in zip(noisy_results, ideal_results):
        fid = compute_hellinger_fidelity(noisy_counts, ideal_counts, SHOTS)
        fidelities.append(fid)

    return {
        "compile_time": compile_time,
        "estimated_qpu_time": estimated_qpu_time,
        "simulated_qpu_time": sim_time,
        "mean_fidelity": float(np.mean(fidelities)),
        "min_fidelity": float(np.min(fidelities)),
        "num_subcircuits": len(flat_circuits),
        "mode": "noisy_simulation",
    }


@bk.foreach(bench=SECONDARY_BENCHMARKS)
@bk.foreach(circuit_size=SECONDARY_SIZES)
@bk.log("logs/hardware/secondary_hardware.jsonl")
def bench_secondary_hardware(bench: str, circuit_size: int) -> dict | None:
    """Run secondary benchmarks on real hardware (if credits permit)."""
    print(f"Hardware [{bench}]: {circuit_size} qubits")

    circuit = get_benchmark(bench, circuit_size).remove_final_measurements(
        inplace=False
    )

    heinsum, flat_circuits, compile_time = compile_qtpu_subcircuits(
        circuit, max_size=QPU_SIZE
    )
    if heinsum is None:
        return None

    estimated_qpu_time = estimate_runtime(flat_circuits)

    backend = get_ibm_backend()
    transpiled = transpile_for_hardware(flat_circuits, backend)
    hw_results, actual_qpu_time = run_on_hardware(transpiled, backend)
    ideal_results = simulate_ideal(flat_circuits)

    fidelities = []
    for hw_counts, ideal_counts in zip(hw_results, ideal_results):
        fid = compute_hellinger_fidelity(hw_counts, ideal_counts, SHOTS)
        fidelities.append(fid)

    return {
        "compile_time": compile_time,
        "estimated_qpu_time": estimated_qpu_time,
        "actual_qpu_time": actual_qpu_time,
        "estimation_error": abs(actual_qpu_time - estimated_qpu_time)
        / max(actual_qpu_time, 1e-9),
        "mean_fidelity": float(np.mean(fidelities)),
        "min_fidelity": float(np.min(fidelities)),
        "num_subcircuits": len(flat_circuits),
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    usage = """
Real IBM Quantum Hardware Benchmark (OSDI Condition 2)

Usage: python -m evaluation.hardware.run <command>

Commands:
    qnn-hardware       Run QNN on real IBM Marrakesh
    qnn-noisy-sim      Run QNN with noisy simulation (fallback)
    secondary          Run secondary benchmarks on hardware
    all-hardware       Run all hardware benchmarks
    all-sim            Run all with noisy simulation fallback

Configuration:
    Backend:       {BACKEND_NAME}
    QPU size:      {QPU_SIZE} qubits
    Shots:         {SHOTS}
    QNN sizes:     {QNN_SIZES}
""".format(
        BACKEND_NAME=BACKEND_NAME,
        QPU_SIZE=QPU_SIZE,
        SHOTS=SHOTS,
        QNN_SIZES=QNN_SIZES,
    )

    if len(sys.argv) < 2:
        print(usage)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "qnn-hardware":
        bench_qnn_hardware()
    elif cmd == "qnn-noisy-sim":
        bench_qnn_noisy_sim()
    elif cmd == "secondary":
        bench_secondary_hardware()
    elif cmd == "all-hardware":
        bench_qnn_hardware()
        bench_secondary_hardware()
    elif cmd == "all-sim":
        bench_qnn_noisy_sim()
    else:
        print(f"Unknown command: {cmd}")
        print(usage)
        sys.exit(1)
