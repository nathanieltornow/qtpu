"""Baseline (naive and batch) execution strategies for HEinsum.

These functions provide inefficient but fair baseline implementations
for comparison with the optimized HEinsum execution. They expand the
ISwitch-based quantum tensors into individual circuits and process them
either sequentially (naive) or in batches.

The key difference from HEinsum's optimized execution:
- Naive/Batch: Generate N separate circuits, compile each individually
- HEinsum: One circuit with ISwitches, compiled once, executed with broadcasting

Both approaches include CUDA-Q code generation for fair comparison.
"""

from __future__ import annotations

from itertools import product
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import torch

from qtpu.compiler.codegen import quantum_tensor_to_cudaq
from qtpu.runtime.timing import TimingBreakdown

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qtpu.core import HEinsum, QuantumTensor


def _expand_iswitch_circuits(qtensor: "QuantumTensor") -> list["QuantumCircuit"]:
    """Expand a QuantumTensor with ISwitches into individual circuits.
    
    For a QuantumTensor with shape (n1, n2, ...), this generates
    n1 * n2 * ... individual circuits, one for each index combination.
    
    Args:
        qtensor: QuantumTensor potentially containing ISwitch instructions.
        
    Returns:
        List of QuantumCircuits (decomposed), one per tensor element.
    """
    # flat() returns circuits with ISwitches bound to specific indices
    # decompose() expands the ISwitches into actual gate sequences
    return [circuit.decompose() for circuit in qtensor.flat()]


def run_naive(
    heinsum: "HEinsum",
    input_tensors: list[torch.Tensor] | None = None,
    circuit_params: dict[str, float] | None = None,
    include_codegen: bool = True,
) -> tuple[torch.Tensor, TimingBreakdown]:
    """Execute HEinsum using naive sequential circuit processing.
    
    This is the worst-case baseline where each circuit is:
    1. Generated/expanded from ISwitches
    2. Code generated for CUDA-Q individually (if include_codegen=True)
    3. Stored for "execution"
    
    This simulates the overhead of not using ISwitch batching.
    
    Args:
        heinsum: The HEinsum to execute.
        input_tensors: Runtime input tensors.
        circuit_params: Circuit parameters (rotation angles, etc.).
        include_codegen: If True, generate CUDA-Q code for each circuit
            (needed for fair comparison with HEinsum which does codegen).
            
    Returns:
        Tuple of (result_tensor, timing_breakdown)
    """
    
    input_tensors = input_tensors or []
    circuit_params = circuit_params or {}
    
    timing = TimingBreakdown(
        device="cpu",
        backend="naive_baseline",
    )
    
    # NOTE: tracemalloc is NOT used here because it significantly slows down
    # CUDA-Q code generation. Memory tracking is done separately if needed.
    
    total_start = perf_counter()
    prep_start = perf_counter()
    
    # Expand all quantum tensors into individual circuits
    all_circuits = []
    circuit_shapes = []  # Track shape for each qtensor
    total_code_lines = 0  # Track total generated code size
    
    for qtensor in heinsum.quantum_tensors:
        circuits = _expand_iswitch_circuits(qtensor)
        circuit_shapes.append((qtensor.shape, len(circuits)))
        
        # Process each circuit ONE BY ONE (naive pattern)
        for i, circuit in enumerate(circuits):
            # Bind parameters if needed
            if circuit.parameters and circuit_params:
                circuit_param_names = {p.name for p in circuit.parameters}
                params_to_bind = {
                    k: v for k, v in circuit_params.items() 
                    if k in circuit_param_names
                }
                if params_to_bind:
                    circuit = circuit.assign_parameters(params_to_bind)
            
            all_circuits.append(circuit)
            
            # Generate CUDA-Q code for this circuit (scalar output)
            # This matches the compilation cost of HEinsum
            if include_codegen:
                _, num_lines = quantum_tensor_to_cudaq(
                    circuit, 
                    shape=(), 
                    kernel_name=f"kernel_naive_{len(all_circuits)}"
                )
                total_code_lines += num_lines
    
    timing.circuit_compilation_time = perf_counter() - prep_start
    timing.total_code_lines = total_code_lines
    
    # For timing comparison, we don't actually execute - just simulate the structure
    # In a real execution, each circuit would be run on QPU
    timing.num_circuits = len(all_circuits)
    
    # Classical contraction (use dummy quantum results)
    contract_start = perf_counter()
    
    # Build dummy quantum results with correct shapes
    quantum_results = []
    for qtensor in heinsum.quantum_tensors:
        # Random values to simulate quantum output
        dummy_result = torch.randn(qtensor.shape, dtype=torch.float64)
        quantum_results.append(dummy_result)
    
    # Get classical tensors
    classical_tensors = [ct.data for ct in heinsum.classical_tensors]
    
    # Contract using einsum
    operands = quantum_results + classical_tensors + list(input_tensors)
    result = torch.einsum(heinsum.einsum_expr, *operands)
    
    timing.classical_contraction_time = perf_counter() - contract_start
    timing.total_time = perf_counter() - total_start
    
    return result, timing


def run_batch(
    heinsum: "HEinsum",
    input_tensors: list[torch.Tensor] | None = None,
    circuit_params: dict[str, float] | None = None,
    include_codegen: bool = True,
) -> tuple[torch.Tensor, TimingBreakdown]:
    """Execute HEinsum using batch circuit processing.
    
    This is a better baseline where:
    1. All circuits are generated/expanded first
    2. Then all are code-generated for CUDA-Q (if include_codegen=True)
    
    This is still less efficient than HEinsum's single-circuit approach
    but better than naive sequential processing.
    
    Args:
        heinsum: The HEinsum to execute.
        input_tensors: Runtime input tensors.
        circuit_params: Circuit parameters (rotation angles, etc.).
        include_codegen: If True, generate CUDA-Q code for each circuit.
            
    Returns:
        Tuple of (result_tensor, timing_breakdown)
    """
    
    input_tensors = input_tensors or []
    circuit_params = circuit_params or {}
    
    timing = TimingBreakdown(
        device="cpu",
        backend="batch_baseline",
    )
    
    # NOTE: tracemalloc is NOT used here because it significantly slows down
    # CUDA-Q code generation. Memory tracking is done separately if needed.
    
    total_start = perf_counter()
    prep_start = perf_counter()
    
    # Step 1: Expand ALL quantum tensors into individual circuits (batched generation)
    all_circuits = []
    circuit_shapes = []
    total_code_lines = 0  # Track total generated code size
    
    for qtensor in heinsum.quantum_tensors:
        circuits = _expand_iswitch_circuits(qtensor)
        circuit_shapes.append((qtensor.shape, len(circuits)))
        
        for circuit in circuits:
            # Bind parameters if needed
            if circuit.parameters and circuit_params:
                circuit_param_names = {p.name for p in circuit.parameters}
                params_to_bind = {
                    k: v for k, v in circuit_params.items() 
                    if k in circuit_param_names
                }
                if params_to_bind:
                    circuit = circuit.assign_parameters(params_to_bind)
            
            all_circuits.append(circuit)
    
    # Step 2: Batch code generation for all circuits
    if include_codegen:
        for i, circuit in enumerate(all_circuits):
            _, num_lines = quantum_tensor_to_cudaq(
                circuit, 
                shape=(), 
                kernel_name=f"kernel_batch_{i}"
            )
            total_code_lines += num_lines
    
    timing.circuit_compilation_time = perf_counter() - prep_start
    timing.total_code_lines = total_code_lines
    timing.num_circuits = len(all_circuits)
    
    # Classical contraction (use dummy quantum results)
    contract_start = perf_counter()
    
    quantum_results = []
    for qtensor in heinsum.quantum_tensors:
        dummy_result = torch.randn(qtensor.shape, dtype=torch.float64)
        quantum_results.append(dummy_result)
    
    classical_tensors = [ct.data for ct in heinsum.classical_tensors]
    operands = quantum_results + classical_tensors + list(input_tensors)
    result = torch.einsum(heinsum.einsum_expr, *operands)
    
    timing.classical_contraction_time = perf_counter() - contract_start
    timing.total_time = perf_counter() - total_start
    
    return result, timing


def run_heinsum(
    heinsum: "HEinsum",
    input_tensors: list[torch.Tensor] | None = None,
    circuit_params: dict[str, float] | None = None,
    skip_execution: bool = True,
) -> tuple[torch.Tensor, TimingBreakdown]:
    """Execute HEinsum using the optimized runtime.
    
    This is the optimized approach:
    1. Single circuit with ISwitches
    2. Compiled once to CUDA-Q with broadcasting
    3. Optionally skip actual execution (for benchmarking compile time only)
    
    Args:
        heinsum: The HEinsum to execute.
        input_tensors: Runtime input tensors.
        circuit_params: Circuit parameters (rotation angles, etc.).
        skip_execution: If True, skip actual quantum simulation and return
            dummy results. This is useful for benchmarking compile time
            without the overhead of simulation for large circuits.
            
    Returns:
        Tuple of (result_tensor, timing_breakdown)
    """
    from qtpu.runtime import HEinsumRuntime, CudaQBackend
    
    
    # Create backend: simulate=False skips actual quantum execution
    backend = CudaQBackend(
        target="qpp-cpu",
        simulate=not skip_execution,
        estimate_qpu_time=True,
    )
    
    runtime = HEinsumRuntime(heinsum, backend=backend, dtype=torch.float64)
    runtime.prepare(optimize=False)
    
    result, timing = runtime.execute(
        input_tensors=input_tensors,
        circuit_params=circuit_params or {},
    )
    
    # Add prep timing and code lines from backend
    if runtime.prep_timing:
        timing.circuit_compilation_time = runtime.prep_timing.circuit_compilation_time
    timing.total_code_lines = backend.total_code_lines
    
    return result, timing


def compare_execution_strategies(
    heinsum: "HEinsum",
    input_tensors: list[torch.Tensor] | None = None,
    circuit_params: dict[str, float] | None = None,
) -> dict[str, TimingBreakdown]:
    """Compare all three execution strategies on the same HEinsum.
    
    Args:
        heinsum: The HEinsum to benchmark.
        input_tensors: Runtime input tensors.
        circuit_params: Circuit parameters.
        
    Returns:
        Dictionary mapping strategy name to TimingBreakdown.
    """
    results = {}
    
    print("Running naive baseline...")
    _, timing = run_naive(heinsum, input_tensors, circuit_params)
    results["naive"] = timing
    print(f"  Circuits: {timing.num_circuits}, Compile time: {timing.circuit_compilation_time:.3f}s")
    
    print("Running batch baseline...")
    _, timing = run_batch(heinsum, input_tensors, circuit_params)
    results["batch"] = timing
    print(f"  Circuits: {timing.num_circuits}, Compile time: {timing.circuit_compilation_time:.3f}s")
    
    print("Running HEinsum (optimized)...")
    _, timing = run_heinsum(heinsum, input_tensors, circuit_params)
    results["heinsum"] = timing
    print(f"  Circuits: {timing.num_circuits}, Compile time: {timing.circuit_compilation_time:.3f}s")
    
    return results
