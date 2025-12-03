"""
Error Mitigation Benchmark: End-to-End Timing & Code Complexity Breakdown
=========================================================================

Compares Naive, Batch (Mitiq-style), and QTPU approaches for error mitigation.

Metrics:
- Preparation time (generation + compilation + CUDA-Q codegen)
- Quantum execution time (estimated for real QPU)
- Classical postprocessing time
- Total code lines generated (measure of compilation complexity)
"""

from __future__ import annotations

import gc
from time import perf_counter

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.circuit.library import IGate, XGate, YGate, ZGate

import benchkit as bk

from evaluation.analysis import estimate_runtime
from qtpu.compiler.codegen import quantum_tensor_to_cudaq
from qtpu.core import ISwitch, QuantumTensor
from qtpu.runtime import CudaQBackend


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
    """Naive approach: generate circuits one at a time, with CUDA-Q codegen."""
    gc.collect()

    pec_indices = get_pec_gate_indices(circuit, num_pec)
    rng = np.random.default_rng(42)

    prep_start = perf_counter()
    total_code_lines = 0

    # Generate all circuits sequentially (naive pattern)
    all_circuits = []
    for i in range(num_samples):
        pec_combo = rng.integers(0, 4, size=len(pec_indices))
        new_circuit = generate_pec_circuit(circuit, pec_indices, pec_combo)
        transpiled = transpile(new_circuit, optimization_level=0)
        all_circuits.append(transpiled)
        
        # Generate CUDA-Q code for this circuit (scalar output, like HEinsum baseline)
        _, num_lines = quantum_tensor_to_cudaq(
            transpiled,
            shape=(),
            kernel_name=f"kernel_naive_{i}",
        )
        total_code_lines += num_lines

    preparation_time = perf_counter() - prep_start

    # Estimate QPU runtime using analysis module
    quantum_time = estimate_runtime(all_circuits)

    # Classical postprocessing (simulated - just the mean computation)
    postproc_start = perf_counter()
    result_values = rng.normal(0.0, 0.5, size=num_samples)
    result_value = float(np.mean(result_values))
    classical_time = perf_counter() - postproc_start

    return {
        "preparation_time": preparation_time,
        "quantum_time": quantum_time,
        "classical_time": classical_time,
        "total_time": preparation_time + quantum_time + classical_time,
        "total_code_lines": total_code_lines,
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
    """Batch approach: generate all circuits upfront, with CUDA-Q codegen."""
    gc.collect()

    pec_indices = get_pec_gate_indices(circuit, num_pec)
    rng = np.random.default_rng(42)

    prep_start = perf_counter()

    # Generate all circuits
    circuits = []
    for _ in range(num_samples):
        pec_combo = rng.integers(0, 4, size=len(pec_indices))
        new_circuit = generate_pec_circuit(circuit, pec_indices, pec_combo)
        circuits.append(new_circuit)

    # Transpile all
    transpiled_circuits = [transpile(c, optimization_level=0) for c in circuits]

    # Generate CUDA-Q code for all circuits (batched)
    total_code_lines = 0
    for i, tc in enumerate(transpiled_circuits):
        _, num_lines = quantum_tensor_to_cudaq(
            tc,
            shape=(),
            kernel_name=f"kernel_batch_{i}",
        )
        total_code_lines += num_lines

    preparation_time = perf_counter() - prep_start

    # Estimate QPU runtime using analysis module
    quantum_time = estimate_runtime(transpiled_circuits)

    # Classical postprocessing
    postproc_start = perf_counter()
    result_values = rng.normal(0.0, 0.5, size=num_samples)
    result_value = float(np.mean(result_values))
    classical_time = perf_counter() - postproc_start

    return {
        "preparation_time": preparation_time,
        "quantum_time": quantum_time,
        "classical_time": classical_time,
        "total_time": preparation_time + quantum_time + classical_time,
        "total_code_lines": total_code_lines,
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
    """QTPU approach: QuantumTensor with CudaQBackend sampling."""
    gc.collect()

    pec_indices = get_pec_gate_indices(circuit, num_pec)

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

    # Create CudaQBackend (simulate=False for benchmarking, estimate_qpu_time=True)
    backend = CudaQBackend(
        target="qpp-cpu",
        simulate=False,
        estimate_qpu_time=True,
    )
    backend.prepare([qtensor])

    preparation_time = perf_counter() - prep_start
    total_code_lines = backend.total_code_lines

    # Sample random indices from the quantum tensor
    rng = np.random.default_rng(42)
    indices = [tuple(rng.integers(0, s) for s in qtensor.shape) for _ in range(num_samples)]
    
    # Sample using backend (returns samples, eval_time, estimated_qpu_time)
    samples, quantum_time, estimated_qpu_time = backend.sample(
        qtensor, indices, params={},
    )

    # Classical postprocessing
    postproc_start = perf_counter()
    values = [val for _, val in samples]
    result_value = float(np.mean(values))
    classical_time = perf_counter() - postproc_start

    # Number of circuits represented by the tensor (4^num_pec)
    num_circuits_represented = int(np.prod(qtensor.shape)) if qtensor.shape else 1

    return {
        "preparation_time": preparation_time,
        "quantum_time": quantum_time,
        "estimated_qpu_time": estimated_qpu_time,
        "classical_time": classical_time,
        "total_time": preparation_time + quantum_time + classical_time,
        "total_code_lines": total_code_lines,
        "result_value": result_value,
        "num_circuits_represented": num_circuits_represented,
        "num_samples": num_samples,
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
