"""
Error Mitigation Benchmark: End-to-End Timing & Memory Breakdown
================================================================

Compares Naive, Batch (Mitiq-style), and QTPU approaches for error mitigation.

Metrics:
- Preparation time (generation + compilation)
- Quantum execution time (estimated for real QPU)
- Classical postprocessing time
- Peak memory during preparation
"""

from __future__ import annotations

import gc
import tracemalloc
from time import perf_counter

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.circuit.library import IGate, XGate, YGate, ZGate

import benchkit as bk

from evaluation.analysis import estimate_runtime
from qtpu.core import ISwitch, QuantumTensor
from qtpu.runtime.backends import FakeQPUCudaQBackend


PAULIS = [IGate(), XGate(), YGate(), ZGate()]


# =============================================================================
# Circuit Helpers
# =============================================================================


def create_test_circuit(num_qubits: int, reps: int = 2) -> QuantumCircuit:
    """Create a test circuit with H, RY, CX gates."""
    qc = QuantumCircuit(num_qubits)
    for layer in range(reps):
        for i in range(num_qubits):
            qc.h(i)
            qc.ry(0.5 * (i + 1) * (layer + 1), i)
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)
    for i in range(num_qubits):
        qc.ry(0.3 * (i + 1), i)
    return qc


def get_pec_gate_indices(circuit: QuantumCircuit, num_pec: int) -> list[int]:
    """Get indices of single-qubit gates to apply PEC."""
    indices = []
    for i, instr in enumerate(circuit.data):
        if len(indices) >= num_pec:
            break
        if instr.operation.num_qubits == 1:
            indices.append(i)
    return indices


def generate_pec_circuit(
    circuit: QuantumCircuit,
    pec_indices: list[int],
    pec_combo: np.ndarray,
) -> QuantumCircuit:
    """Generate a single PEC circuit variant."""
    new_circuit = QuantumCircuit(circuit.num_qubits)
    pec_count = 0
    for idx, instr in enumerate(circuit.data):
        if idx in pec_indices:
            new_circuit.append(PAULIS[pec_combo[pec_count]], instr.qubits)
            new_circuit.append(instr.operation, instr.qubits)
            pec_count += 1
        else:
            new_circuit.append(instr.operation, instr.qubits, instr.clbits)
    return new_circuit


# =============================================================================
# NAIVE Approach (Sequential)
# =============================================================================


def run_naive(
    circuit: QuantumCircuit,
    num_pec: int,
    num_samples: int,
) -> dict:
    """Naive approach: generate circuits one at a time, estimate QPU runtime."""
    gc.collect()

    pec_indices = get_pec_gate_indices(circuit, num_pec)
    rng = np.random.default_rng(42)

    # Track memory during preparation (first circuit only for naive)
    tracemalloc.start()
    prep_start = perf_counter()

    # Build first circuit to measure prep time
    pec_combo = rng.integers(0, 4, size=len(pec_indices))
    first_circuit = generate_pec_circuit(circuit, pec_indices, pec_combo)
    first_transpiled = transpile(first_circuit, optimization_level=0)

    preparation_time = perf_counter() - prep_start
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Generate all remaining circuits for QPU time estimation
    gen_start = perf_counter()
    all_circuits = [first_transpiled]
    for _ in range(num_samples - 1):
        pec_combo = rng.integers(0, 4, size=len(pec_indices))
        new_circuit = generate_pec_circuit(circuit, pec_indices, pec_combo)
        all_circuits.append(transpile(new_circuit, optimization_level=0))
    generation_time = perf_counter() - gen_start

    # Estimate QPU runtime using analysis module
    quantum_time = estimate_runtime(all_circuits)

    # Classical postprocessing (simulated - just the mean computation)
    postproc_start = perf_counter()
    # Simulate result values (in reality these come from QPU)
    result_values = rng.normal(0.0, 0.5, size=num_samples)
    result_value = float(np.mean(result_values))
    classical_time = perf_counter() - postproc_start

    return {
        "preparation_time": preparation_time,
        "generation_time": generation_time,
        "quantum_time": quantum_time,
        "classical_time": classical_time,
        "total_time": preparation_time + generation_time + quantum_time + classical_time,
        "peak_memory": peak_memory,
        "result_value": result_value,
        "num_circuits": num_samples,
    }


# =============================================================================
# BATCH (Mitiq-style) Approach
# =============================================================================


def run_batch(
    circuit: QuantumCircuit,
    num_pec: int,
    num_samples: int,
) -> dict:
    """Batch approach: generate all circuits upfront, estimate QPU runtime."""
    gc.collect()

    pec_indices = get_pec_gate_indices(circuit, num_pec)
    rng = np.random.default_rng(42)

    # Track memory during preparation
    tracemalloc.start()
    prep_start = perf_counter()

    # Generate all circuits
    circuits = []
    for _ in range(num_samples):
        pec_combo = rng.integers(0, 4, size=len(pec_indices))
        new_circuit = generate_pec_circuit(circuit, pec_indices, pec_combo)
        circuits.append(new_circuit)

    # Transpile all
    transpiled_circuits = [transpile(c, optimization_level=0) for c in circuits]

    preparation_time = perf_counter() - prep_start
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Estimate QPU runtime using analysis module
    quantum_time = estimate_runtime(transpiled_circuits)

    # Classical postprocessing
    postproc_start = perf_counter()
    # Simulate result values (in reality these come from QPU)
    result_values = rng.normal(0.0, 0.5, size=num_samples)
    result_value = float(np.mean(result_values))
    classical_time = perf_counter() - postproc_start

    return {
        "preparation_time": preparation_time,
        "generation_time": 0.0,  # Included in preparation
        "quantum_time": quantum_time,
        "classical_time": classical_time,
        "total_time": preparation_time + quantum_time + classical_time,
        "peak_memory": peak_memory,
        "result_value": result_value,
        "num_circuits": num_samples,
    }


# =============================================================================
# QTPU Approach
# =============================================================================


def create_pec_iswitch(idx: str) -> ISwitch:
    """Create ISwitch for PEC basis operations."""
    param = Parameter(idx)

    def selector(basis_idx: int) -> QuantumCircuit:
        qc = QuantumCircuit(1)
        qc.append(PAULIS[basis_idx], [0])
        return qc

    return ISwitch(param, 1, 4, selector)


def run_qtpu(
    circuit: QuantumCircuit,
    num_pec: int,
    num_samples: int,
) -> dict:
    """QTPU approach: QuantumTensor with FakeQPUCudaQBackend."""
    gc.collect()

    pec_indices = get_pec_gate_indices(circuit, num_pec)

    # Track memory during preparation
    tracemalloc.start()
    prep_start = perf_counter()

    # Build circuit with ISwitches
    em_circuit = QuantumCircuit(circuit.num_qubits)
    pec_count = 0
    for idx, instr in enumerate(circuit.data):
        if idx in pec_indices:
            iswitch = create_pec_iswitch(f"pec_{pec_count}")
            em_circuit.append(iswitch, instr.qubits)
            em_circuit.append(instr.operation, instr.qubits)
            pec_count += 1
        else:
            em_circuit.append(instr.operation, instr.qubits, instr.clbits)

    # Create QuantumTensor
    qtensor = QuantumTensor(em_circuit)

    # Create FakeQPUCudaQBackend and prepare
    backend = FakeQPUCudaQBackend(
        backend_name="FakeMarrakesh",
        target="qpp-cpu",
        shots=1000,
        optimization_level=3,
        warmup=True,
    )
    backend.prepare([qtensor])

    preparation_time = perf_counter() - prep_start
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Get the compiled tensor
    compiled = backend._compiled_cache[id(qtensor)]

    # Sample (quantum execution)
    exec_start = perf_counter()
    samples = compiled.sample(num_samples)
    quantum_time = perf_counter() - exec_start

    # Get estimated QPU time from backend (this is for ALL circuits in the tensor)
    # Scale it by the fraction of circuits actually sampled
    num_circuits_represented = int(np.prod(qtensor.shape)) if qtensor.shape else 1
    full_qpu_time = backend._qpu_time_cache.get(id(qtensor), 0.0)
    # QTPU only runs num_samples circuits, not all num_circuits_represented
    estimated_qpu_time = full_qpu_time * (num_samples / num_circuits_represented)

    # Classical postprocessing
    postproc_start = perf_counter()
    values = [val for _, val in samples]
    result_value = float(np.mean(values))
    classical_time = perf_counter() - postproc_start

    return {
        "preparation_time": preparation_time,
        "generation_time": 0.0,  # Included in preparation
        "quantum_time": quantum_time,
        "estimated_qpu_time": estimated_qpu_time,
        "classical_time": classical_time,
        "total_time": preparation_time + quantum_time + classical_time,
        "peak_memory": peak_memory,
        "result_value": result_value,
        "num_circuits_represented": num_circuits_represented,
    }


# =============================================================================
# Benchmark Configuration
# =============================================================================

CIRCUIT_SIZES = [4, 6, 8, 10, 12]
NUM_SAMPLES_LIST = [100, 500, 1000, 10000]


@bk.foreach(circuit_size=CIRCUIT_SIZES)
@bk.foreach(num_samples=NUM_SAMPLES_LIST)
@bk.log("logs/error_mitigation/naive_breakdown.jsonl")
def bench_naive(circuit_size: int, num_samples: int) -> dict:
    """Benchmark naive (sequential) approach."""
    circuit = create_test_circuit(circuit_size, reps=2)
    num_pec = circuit_size
    return run_naive(circuit, num_pec, num_samples)


@bk.foreach(circuit_size=CIRCUIT_SIZES)
@bk.foreach(num_samples=NUM_SAMPLES_LIST)
@bk.log("logs/error_mitigation/batch_breakdown.jsonl")
def bench_batch(circuit_size: int, num_samples: int) -> dict:
    """Benchmark batch (Mitiq-style) approach."""
    circuit = create_test_circuit(circuit_size, reps=2)
    num_pec = circuit_size
    return run_batch(circuit, num_pec, num_samples)


@bk.foreach(circuit_size=CIRCUIT_SIZES)
@bk.foreach(num_samples=NUM_SAMPLES_LIST)
@bk.log("logs/error_mitigation/qtpu_breakdown.jsonl")
def bench_qtpu(circuit_size: int, num_samples: int) -> dict:
    """Benchmark QTPU approach."""
    circuit = create_test_circuit(circuit_size, reps=2)
    num_pec = circuit_size
    return run_qtpu(circuit, num_pec, num_samples)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python bench_breakdown.py [naive|batch|qtpu|all]")
        sys.exit(1)

    if sys.argv[1] == "naive":
        bench_naive()
    elif sys.argv[1] == "batch":
        bench_batch()
    elif sys.argv[1] == "qtpu":
        bench_qtpu()
    elif sys.argv[1] == "all":
        bench_naive()
        bench_batch()
        bench_qtpu()
    else:
        print(f"Unknown option: {sys.argv[1]}")
        sys.exit(1)
