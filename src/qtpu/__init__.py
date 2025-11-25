"""Quantum Tensor Processing Unit (QTPU) - A framework for scalable quantum-classical computation."""

from __future__ import annotations

from .compiler import cut
from .contract import contract, evaluate, execute, get_quasi_probability, sample
from .transforms import circuit_to_hybrid_tn

__all__ = [
    "circuit_to_hybrid_tn",
    "contract",
    "cut",
    "evaluate",
    "execute",
    "get_quasi_probability",
    "sample",
]

# Lazy import for torch module (optional dependency)
def __getattr__(name: str):
    if name == "torch":
        from qtpu import torch as torch_module
        return torch_module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
