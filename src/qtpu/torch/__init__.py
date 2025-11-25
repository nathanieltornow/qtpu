"""PyTorch integration for hybrid tensor networks."""

from __future__ import annotations

from qtpu.torch.evaluator import (
    ExpvalTorchEvaluator,
    StatevectorTorchEvaluator,
    TorchCircuitEvaluator,
)
from qtpu.torch.htn_layer import (
    HTNLayer,
    ParameterizedCircuitTensor,
)
from qtpu.torch.differentiable import (
    DifferentiableTorchEvaluator,
    differentiable_evaluate,
)

__all__ = [
    # Evaluators
    "TorchCircuitEvaluator",
    "ExpvalTorchEvaluator",
    "StatevectorTorchEvaluator",
    "DifferentiableTorchEvaluator",
    # Layers
    "HTNLayer",
    "ParameterizedCircuitTensor",
    # Functions
    "differentiable_evaluate",
]
