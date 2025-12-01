"""Differentiable HEinsum contraction runtime with multiple backend support.

This module provides a clean, extensible runtime for hybrid tensor network
contraction with support for:
- Multiple quantum backends (simulator, fake QPU timing estimation, CUDA-Q, real QPU)
- Detailed timing breakdowns for evaluation
- GPU acceleration via PyTorch
- JIT compilation for maximum performance

Example:
    >>> from qtpu.runtime import HEinsumRuntime, FakeQPUBackend
    >>> 
    >>> # Create runtime with fake QPU (estimates timing, returns random results)
    >>> runtime = HEinsumRuntime(heinsum, backend="fake_qpu")
    >>> runtime.prepare(jit=True)
    >>> 
    >>> # Execute with timing breakdown
    >>> result, timing = runtime.execute(input_tensors=[x])
    >>> print(f"Quantum: {timing.quantum_time:.3f}s, Classical: {timing.classical_time:.3f}s")
    
    >>> # Or use CUDA-Q for GPU-accelerated quantum simulation
    >>> runtime = HEinsumRuntime(heinsum, backend="cudaq-nvidia")
    >>> result, timing = runtime.execute(input_tensors=[x])
"""

from qtpu.runtime.timing import (
    TimingBreakdown,
    AggregateTimingStats,
)
from qtpu.runtime.backends import (
    QuantumBackend,
    SimulatorBackend,
    FakeQPUBackend,
    CudaQBackend,
)
from qtpu.runtime.device import (
    Device,
    get_device,
)
from qtpu.runtime.executor import (
    HEinsumRuntime,
    HEinsumContractor,
)
# Legacy aliases
QuantumTensorEvaluator = SimulatorBackend

__all__ = [
    # Timing
    "TimingBreakdown",
    "AggregateTimingStats",
    # Backends
    "QuantumBackend",
    "SimulatorBackend",
    "FakeQPUBackend",
    "CudaQBackend",
    "QuantumTensorEvaluator",  # Legacy alias
    # Device
    "Device",
    "get_device",
    # Executor
    "HEinsumRuntime",
    "HEinsumContractor",
]
