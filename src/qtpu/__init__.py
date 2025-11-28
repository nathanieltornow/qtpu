"""Quantum Tensor Processing Unit (QTPU) - A framework for scalable quantum-classical computation."""

from __future__ import annotations

from .compiler import cut
from .transforms import circuit_to_heinsum
from .runtime import (
    HEinsumRuntime,
    HEinsumContractor,  # Backward compatibility
    SimulatorBackend,
    FakeQPUBackend,
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
    # Runtime
    "HEinsumRuntime",
    "HEinsumContractor",
    "SimulatorBackend",
    "FakeQPUBackend",
    "QuantumBackend",
    "TimingBreakdown",
    "AggregateTimingStats",
    "Device",
    "get_device",
]
