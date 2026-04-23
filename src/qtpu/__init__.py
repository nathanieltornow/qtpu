"""Quantum Tensor Processing Unit (QTPU) - A framework for scalable quantum-classical computation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .compiler import cut as compile
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
    HEinsumContractor,
    QuantumBackend,
    TimingBreakdown,
    AggregateTimingStats,
    Device,
    get_device,
)

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


def compile_to_heinsum(
    circuit: QuantumCircuit,
    max_size: int | None = None,
    max_c_cost: float | None = None,
    cost_weight: float = 1.0,
    num_workers: int | None = None,
    n_trials: int = 100,
    seed: int | None = None,
) -> HEinsum:
    """Compile a quantum circuit into a hybrid tensor network (hTN).

    Convenience wrapper that calls ``compile`` followed by ``circuit_to_heinsum``.

    Args:
        circuit: The quantum circuit to compile.
        max_size: Maximum subcircuit width (qubits).
        max_c_cost: Maximum classical cost.
        cost_weight: Weight for classical cost vs error reduction.
        num_workers: Number of parallel workers.
        n_trials: Number of optimization trials.
        seed: Random seed.

    Returns:
        The compiled HEinsum (hybrid tensor network).
    """
    cut_circuit = compile(
        circuit,
        max_size=max_size,
        max_c_cost=max_c_cost,
        cost_weight=cost_weight,
        num_workers=num_workers,
        n_trials=n_trials,
        seed=seed,
    )
    return circuit_to_heinsum(cut_circuit)


cut = compile  # backward compatibility

__all__ = [
    # Compiler
    "compile",
    "compile_to_heinsum",
    "cut",  # backward compatibility
    "circuit_to_heinsum",
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
