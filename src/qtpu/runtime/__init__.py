"""Differentiable HEinsum contraction runtime.

This module provides a clean runtime for hybrid tensor network contraction:
- CUDA-Q backend with optional simulation and QPU time estimation
- Detailed timing breakdowns for evaluation
- GPU acceleration via PyTorch
- JIT compilation for maximum performance

Example:
    >>> from qtpu.runtime import HEinsumRuntime, CudaQBackend
    >>> 
    >>> # Full simulation mode
    >>> runtime = HEinsumRuntime(heinsum, backend="cudaq")
    >>> runtime.prepare()
    >>> result, timing = runtime.execute(input_tensors=[x])
    >>> 
    >>> # Benchmarking mode (no simulation, just timing estimation)
    >>> backend = CudaQBackend(simulate=False, estimate_qpu_time=True)
    >>> runtime = HEinsumRuntime(heinsum, backend=backend)
    >>> result, timing = runtime.execute(input_tensors=[x])
    >>> # timing.quantum_estimated_qpu_time = estimated real QPU time
"""

from qtpu.runtime.timing import (
    TimingBreakdown,
    AggregateTimingStats,
)
from qtpu.runtime.backends import (
    QuantumBackend,
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
from qtpu.runtime.baseline import (
    run_naive,
    run_batch,
    run_heinsum,
    compare_execution_strategies,
)

__all__ = [
    # Timing
    "TimingBreakdown",
    "AggregateTimingStats",
    # Backends
    "QuantumBackend",
    "CudaQBackend",
    # Device
    "Device",
    "get_device",
    # Executor
    "HEinsumRuntime",
    "HEinsumContractor",
    # Baseline execution strategies
    "run_naive",
    "run_batch",
    "run_heinsum",
    "compare_execution_strategies",
]
