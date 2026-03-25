"""
Benchmark: Error Mitigation Host Overhead Comparison
====================================================

Compares three approaches for error mitigation circuit generation and execution:

1. NAIVE LOOP:
   For each sample: generate circuit → transpile → execute → store
   Most straightforward but slowest approach.

2. BATCH:
   Generate all circuits → Transpile all → Batch execute → Process results
   Common optimization pattern used by Mitiq-style tools.

3. QTPU (CompiledQuantumTensor + CudaQ):
   Define circuit with ISwitches → JIT compile once → Sample from index space
   Our approach with JIT compilation and CUDA-Q broadcasting.

Metrics tracked per approach:
- Circuit generation time (CPU)
- Circuit compilation/transpilation time (CPU)  
- Quantum execution time (simulator)
- Classical post-processing time
- Total preparation time (gen + compile)
- Total end-to-end time

Mitigation types supported:
- PEC (Probabilistic Error Cancellation)
- Pauli Twirling
- Combined (PEC + Twirling)
"""

from __future__ import annotations

import tracemalloc
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter, ClassicalRegister
from qiskit.circuit.library import IGate, XGate, YGate, ZGate
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

from evaluation.utils import log_result

from qtpu.core import ISwitch, QuantumTensor

if TYPE_CHECKING:
    pass


PAULIS = [IGate(), XGate(), YGate(), ZGate()]
PAULI_LABELS = ["I", "X", "Y", "Z"]


# =============================================================================
# Circuit Generation Helpers
# =============================================================================


def get_twirl_positions(circuit: QuantumCircuit, num_twirl: int) -> list[int]:
    """Get qubit indices to apply twirling."""
    return list(range(min(num_twirl, circuit.num_qubits)))


def get_pec_gate_indices(circuit: QuantumCircuit, num_pec: int) -> list[int]:
    """Get indices of single-qubit gates to apply PEC."""
    indices = []
    for i, instr in enumerate(circuit.data):
        if len(indices) >= num_pec:
            break
        if instr.operation.num_qubits == 1:
            indices.append(i)
    return indices


# =============================================================================
# 1. NAIVE LOOP - Generate, transpile, execute one at a time
# =============================================================================


def run_naive_loop(
    circuit: QuantumCircuit,
    mitigation: str,
    num_twirl: int,
    num_pec: int,
    num_samples: int,
) -> dict:
    """Naive approach: loop over all samples, generate/execute each circuit."""

    total_start = perf_counter()
    
    estimator = StatevectorEstimator()
    observable = SparsePauliOp.from_list([("Z" * circuit.num_qubits, 1.0)])
    
    twirl_positions = get_twirl_positions(circuit, num_twirl)
    pec_indices = get_pec_gate_indices(circuit, num_pec)
    
    rng = np.random.default_rng(42)
    results = []
    
    gen_time = 0.0
    compile_time = 0.0
    exec_time = 0.0
    
    for sample_idx in range(num_samples):
        # Generate circuit for this sample
        gen_start = perf_counter()
        
        new_circuit = QuantumCircuit(circuit.num_qubits)
        
        if mitigation in ["twirl", "combined"]:
            # Pre-twirl: random Paulis
            twirl_combo = rng.integers(0, 4, size=len(twirl_positions))
            for i, pauli_idx in enumerate(twirl_combo):
                new_circuit.append(PAULIS[pauli_idx], [twirl_positions[i]])
        
        # Main circuit with PEC insertions
        pec_combo = rng.integers(0, 4, size=len(pec_indices)) if mitigation in ["pec", "combined"] else []
        pec_count = 0
        
        for idx, instr in enumerate(circuit.data):
            if mitigation in ["pec", "combined"] and idx in pec_indices:
                new_circuit.append(PAULIS[pec_combo[pec_count]], instr.qubits)
                new_circuit.append(instr.operation, instr.qubits)
                pec_count += 1
            else:
                new_circuit.append(instr.operation, instr.qubits, instr.clbits)
        
        if mitigation in ["twirl", "combined"]:
            # Post-twirl: same Paulis (they cancel for Clifford gates)
            for i, pauli_idx in enumerate(twirl_combo):
                new_circuit.append(PAULIS[pauli_idx], [twirl_positions[i]])
        
        gen_time += perf_counter() - gen_start
        
        # Transpile
        compile_start = perf_counter()
        transpiled = transpile(new_circuit, optimization_level=0)
        compile_time += perf_counter() - compile_start
        
        # Execute
        exec_start = perf_counter()
        job = estimator.run([(transpiled, observable)])
        exp_val = job.result()[0].data.evs
        exec_time += perf_counter() - exec_start
        
        results.append(exp_val)
    
    # Classical post-processing
    post_start = perf_counter()
    final_result = np.mean(results)
    classical_time = perf_counter() - post_start
    
    preparation_time = gen_time + compile_time
    total_time = perf_counter() - total_start
    
    return {
        "generation_time": gen_time,
        "compilation_time": compile_time,
        "preparation_time": preparation_time,
        "quantum_time": exec_time,
        "classical_time": classical_time,
        "total_time": total_time,
        "num_circuits": num_samples,
    }


# =============================================================================
# 2. BATCH - Generate all, transpile all, execute all
# =============================================================================


def run_batch(
    circuit: QuantumCircuit,
    mitigation: str,
    num_twirl: int,
    num_pec: int,
    num_samples: int,
) -> dict:
    """Batch approach: generate all circuits, then batch execute."""

    total_start = perf_counter()
    
    twirl_positions = get_twirl_positions(circuit, num_twirl)
    pec_indices = get_pec_gate_indices(circuit, num_pec)
    
    rng = np.random.default_rng(42)
    
    # Generate all circuits
    tracemalloc.start()
    gen_start = perf_counter()
    circuits = []
    
    for sample_idx in range(num_samples):
        new_circuit = QuantumCircuit(circuit.num_qubits)
        
        if mitigation in ["twirl", "combined"]:
            twirl_combo = rng.integers(0, 4, size=len(twirl_positions))
            for i, pauli_idx in enumerate(twirl_combo):
                new_circuit.append(PAULIS[pauli_idx], [twirl_positions[i]])
        
        pec_combo = rng.integers(0, 4, size=len(pec_indices)) if mitigation in ["pec", "combined"] else []
        pec_count = 0
        
        for idx, instr in enumerate(circuit.data):
            if mitigation in ["pec", "combined"] and idx in pec_indices:
                new_circuit.append(PAULIS[pec_combo[pec_count]], instr.qubits)
                new_circuit.append(instr.operation, instr.qubits)
                pec_count += 1
            else:
                new_circuit.append(instr.operation, instr.qubits, instr.clbits)
        
        if mitigation in ["twirl", "combined"]:
            for i, pauli_idx in enumerate(twirl_combo):
                new_circuit.append(PAULIS[pauli_idx], [twirl_positions[i]])
        
        circuits.append(new_circuit)
    
    generation_time = perf_counter() - gen_start
    _, generation_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Transpile all
    compile_start = perf_counter()
    transpiled_circuits = transpile(circuits, optimization_level=0)
    compilation_time = perf_counter() - compile_start
    
    preparation_time = generation_time + compilation_time
    
    # Batch execute
    exec_start = perf_counter()
    estimator = StatevectorEstimator()
    observable = SparsePauliOp.from_list([("Z" * circuit.num_qubits, 1.0)])
    
    jobs = [(qc, observable) for qc in transpiled_circuits]
    batch_result = estimator.run(jobs).result()
    exp_vals = [r.data.evs for r in batch_result]
    quantum_time = perf_counter() - exec_start
    
    # Classical post-processing
    post_start = perf_counter()
    final_result = np.mean(exp_vals)
    classical_time = perf_counter() - post_start
    
    total_time = perf_counter() - total_start
    
    return {
        "generation_time": generation_time,
        "compilation_time": compilation_time,
        "preparation_time": preparation_time,
        "generation_memory": generation_memory,
        "quantum_time": quantum_time,
        "classical_time": classical_time,
        "total_time": total_time,
        "num_circuits": num_samples,
    }


# =============================================================================
# 3. QTPU (CompiledQuantumTensor + CudaQ) - ISwitch + sample from index space
# =============================================================================


def create_twirl_iswitch(idx: str) -> ISwitch:
    """Create ISwitch for Pauli twirling."""
    param = Parameter(idx)
    
    def selector(pauli_idx: int) -> QuantumCircuit:
        qc = QuantumCircuit(1)
        qc.append(PAULIS[pauli_idx], [0])
        return qc
    
    return ISwitch(param, 1, 4, selector)


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
    mitigation: str,
    num_twirl: int,
    num_pec: int,
    num_samples: int,
) -> dict:
    """QTPU approach: CompiledQuantumTensor with sample() method."""

    total_start = perf_counter()
    
    twirl_positions = get_twirl_positions(circuit, num_twirl)
    pec_indices = get_pec_gate_indices(circuit, num_pec)
    
    # Build circuit with ISwitches
    tracemalloc.start()
    gen_start = perf_counter()
    
    em_circuit = QuantumCircuit(circuit.num_qubits)
    em_circuit.add_register(ClassicalRegister(circuit.num_qubits))
    
    # Pre-twirl ISwitches
    if mitigation in ["twirl", "combined"]:
        for i in twirl_positions:
            iswitch = create_twirl_iswitch(f"twirl_pre_{i}")
            em_circuit.append(iswitch, [i])
    
    # Main circuit with PEC ISwitches
    pec_count = 0
    for idx, instr in enumerate(circuit.data):
        if mitigation in ["pec", "combined"] and idx in pec_indices:
            iswitch = create_pec_iswitch(f"pec_{pec_count}")
            em_circuit.append(iswitch, instr.qubits)
            em_circuit.append(instr.operation, instr.qubits)
            pec_count += 1
        else:
            em_circuit.append(instr.operation, instr.qubits, instr.clbits)
    
    # Post-twirl ISwitches
    if mitigation in ["twirl", "combined"]:
        for i in twirl_positions:
            iswitch = create_twirl_iswitch(f"twirl_post_{i}")
            em_circuit.append(iswitch, [i])
    
    # Add measurement
    em_circuit.measure(range(circuit.num_qubits), range(circuit.num_qubits))
    
    # Create QuantumTensor
    qtensor = QuantumTensor(em_circuit)
    
    generation_time = perf_counter() - gen_start
    _, generation_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Count total represented circuits
    num_circuits_represented = int(np.prod(qtensor.shape)) if qtensor.shape else 1
    
    # Compile to CUDA-Q (includes JIT warmup)
    compile_start = perf_counter()
    compiled = qtensor.compile("cudaq")  # Warmup=True by default
    compilation_time = perf_counter() - compile_start
    
    preparation_time = generation_time + compilation_time
    
    # Sample from index space using CUDA-Q broadcasting
    exec_start = perf_counter()
    samples = compiled.sample(num_samples)
    quantum_time = perf_counter() - exec_start
    
    # Classical post-processing (compute weighted average)
    post_start = perf_counter()
    values = [val for _, val in samples]
    final_result = np.mean(values)
    classical_time = perf_counter() - post_start
    
    total_time = perf_counter() - total_start
    
    return {
        "generation_time": generation_time,
        "compilation_time": compilation_time,
        "preparation_time": preparation_time,
        "generation_memory": generation_memory,
        "quantum_time": quantum_time,
        "classical_time": classical_time,
        "total_time": total_time,
        "num_samples": num_samples,
        "num_circuits_represented": num_circuits_represented,
    }


# =============================================================================
# BENCHMARK CONFIGURATION
# =============================================================================

CIRCUIT_SIZES = [4, 6, 8, 10]
MITIGATIONS = ["pec", "twirl", "combined"]
NUM_SAMPLES_LIST = [100, 1000, 10000]


def get_mitigation_params(circuit_size: int) -> tuple[int, int]:
    """Get (num_twirl, num_pec) based on circuit size."""
    return circuit_size, circuit_size


def create_test_circuit(num_qubits: int, reps: int = 2) -> QuantumCircuit:
    """Create a test circuit with only basic gates (H, RY, CX).
    
    This avoids gates like 'r' and 'p' that aren't in CUDA-Q's gate map.
    """
    qc = QuantumCircuit(num_qubits)
    
    for layer in range(reps):
        # Rotation layer (Hadamard + RY)
        for i in range(num_qubits):
            qc.h(i)
            qc.ry(0.5 * (i + 1) * (layer + 1), i)
        
        # Entanglement layer (linear CX chain)
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)
    
    # Final rotation layer
    for i in range(num_qubits):
        qc.ry(0.3 * (i + 1), i)
    
    return qc


def bench_naive_overhead(
    circuit_size: int, mitigation: str, num_samples: int
) -> dict:
    """Benchmark naive loop approach."""
    circuit = create_test_circuit(circuit_size, reps=2)
    num_twirl, num_pec = get_mitigation_params(circuit_size)
    return run_naive_loop(circuit, mitigation, num_twirl, num_pec, num_samples)


def bench_batch_overhead(
    circuit_size: int, mitigation: str, num_samples: int
) -> dict:
    """Benchmark batch approach."""
    circuit = create_test_circuit(circuit_size, reps=2)
    num_twirl, num_pec = get_mitigation_params(circuit_size)
    return run_batch(circuit, mitigation, num_twirl, num_pec, num_samples)


def bench_qtpu_overhead(
    circuit_size: int, mitigation: str, num_samples: int
) -> dict:
    """Benchmark QTPU CompiledQuantumTensor with sample() method."""
    circuit = create_test_circuit(circuit_size, reps=2)
    num_twirl, num_pec = get_mitigation_params(circuit_size)
    return run_qtpu(circuit, mitigation, num_twirl, num_pec, num_samples)


def _run_sweep(bench_fn, log_path):
    for circuit_size in CIRCUIT_SIZES:
        for mitigation in MITIGATIONS:
            for num_samples in NUM_SAMPLES_LIST:
                config = {"circuit_size": circuit_size, "mitigation": mitigation, "num_samples": num_samples}
                print(f"  Config: {config}")
                result = bench_fn(circuit_size, mitigation, num_samples)
                log_result(log_path, config, result)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python run_overhead.py [naive|batch|qtpu|all]")
        sys.exit(1)

    if sys.argv[1] == "naive":
        _run_sweep(bench_naive_overhead, "logs/error_mitigation/naive_overhead.jsonl")
    elif sys.argv[1] == "batch":
        _run_sweep(bench_batch_overhead, "logs/error_mitigation/batch_overhead.jsonl")
    elif sys.argv[1] == "qtpu":
        _run_sweep(bench_qtpu_overhead, "logs/error_mitigation/qtpu_overhead.jsonl")
    elif sys.argv[1] == "all":
        _run_sweep(bench_naive_overhead, "logs/error_mitigation/naive_overhead.jsonl")
        _run_sweep(bench_batch_overhead, "logs/error_mitigation/batch_overhead.jsonl")
        _run_sweep(bench_qtpu_overhead, "logs/error_mitigation/qtpu_overhead.jsonl")
