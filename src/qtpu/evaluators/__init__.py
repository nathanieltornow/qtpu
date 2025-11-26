"""Evaluator classes for evaluating quantum tensors to classical tensors."""

from __future__ import annotations

from ._estimator import ExpvalEvaluator
from ._evaluator import CircuitTensorEvaluator
from ._sampler import SamplerEvaluator
from ._backend import BackendEvaluator

try:
    from ._torch_evaluator import TorchQuantumTensorEvaluator, DifferentiableTorchEvaluator
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    TorchQuantumTensorEvaluator = None
    DifferentiableTorchEvaluator = None

__all__ = [
    "CircuitTensorEvaluator",
    "ExpvalEvaluator",
    "SamplerEvaluator",
    "BackendEvaluator",
]

if _TORCH_AVAILABLE:
    __all__.extend([
        "TorchQuantumTensorEvaluator",
        "DifferentiableTorchEvaluator",
    ])
