"""Quantum Tensor Processing Unit (QTPU) - A framework for scalable quantum-classical computation."""

from __future__ import annotations

from .compiler import cut
from .transforms import circuit_to_heinsum
from .core import (
    QuantumTensor,
    ISwitch,
    CTensor,
    TensorSpec,
    HEinsum,
    rand_regular_heinsum,
)
from .runtime import (
    HEinsumRuntime,
    HEinsumContractor,  # Backward compatibility
    QuantumBackend,
    TimingBreakdown,
    AggregateTimingStats,
    Device,
    get_device,
)

__all__ = [
    # Core
    "circuit_to_heinsum",
    "cut",
    # Tensor
    "QuantumTensor",
    "ISwitch",
    "CTensor",
    "TensorSpec",
    "HEinsum",
    "rand_regular_heinsum",
    # Runtime
    "HEinsumRuntime",
    "HEinsumContractor",
    "QuantumBackend",
    "TimingBreakdown",
    "AggregateTimingStats",
    "Device",
    "get_device",
]
