"""Differentiable circuit evaluation with parameter-shift gradients."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.autograd import Function

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit

    from qtpu.torch.evaluator import TorchCircuitEvaluator


class ParameterShiftFunction(Function):
    """Autograd function implementing parameter-shift rule for quantum gradients.
    
    The parameter-shift rule states that for a parameterized gate R(θ):
        ∂/∂θ <ψ|U†(θ) O U(θ)|ψ> = (1/2)[f(θ + π/2) - f(θ - π/2)]
    
    where f(θ) is the expectation value at parameter θ.
    """

    @staticmethod
    def forward(
        ctx,
        circuit: QuantumCircuit,
        evaluator: TorchCircuitEvaluator,
        param_names: list[str],
        *param_values: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: evaluate circuit with current parameters.

        Args:
            ctx: Autograd context for saving tensors.
            circuit: The quantum circuit template.
            evaluator: The circuit evaluator.
            param_names: Names of parameters in order.
            *param_values: Parameter values as tensors.

        Returns:
            torch.Tensor: Evaluation result.
        """
        # Save for backward
        ctx.circuit = circuit
        ctx.evaluator = evaluator
        ctx.param_names = param_names
        ctx.param_values = param_values
        
        # Build param dict
        params = dict(zip(param_names, param_values, strict=False))
        
        # Evaluate
        result = evaluator.evaluate(circuit, params)
        
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass: compute gradients using parameter-shift rule.

        Args:
            ctx: Autograd context with saved tensors.
            grad_output: Gradient of the loss w.r.t. output.

        Returns:
            Tuple of gradients (None for circuit, evaluator, param_names, then param gradients).
        """
        circuit = ctx.circuit
        evaluator = ctx.evaluator
        param_names = ctx.param_names
        param_values = ctx.param_values
        
        shift = np.pi / 2
        grads = []
        
        for i, (name, value) in enumerate(zip(param_names, param_values)):
            # Shifted parameters
            shifted_plus = list(param_values)
            shifted_plus[i] = value + shift
            
            shifted_minus = list(param_values)
            shifted_minus[i] = value - shift
            
            params_plus = dict(zip(param_names, shifted_plus, strict=False))
            params_minus = dict(zip(param_names, shifted_minus, strict=False))
            
            # Evaluate at shifted points
            result_plus = evaluator.evaluate(circuit, params_plus)
            result_minus = evaluator.evaluate(circuit, params_minus)
            
            # Parameter-shift gradient
            grad = 0.5 * (result_plus - result_minus)
            grads.append(grad_output * grad)
        
        # Return: None for circuit, evaluator, param_names; then param gradients
        return (None, None, None, *grads)


def differentiable_evaluate(
    circuit: QuantumCircuit,
    evaluator: TorchCircuitEvaluator,
    params: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Evaluate a circuit with differentiable parameters.

    This function wraps circuit evaluation in a custom autograd function
    that implements the parameter-shift rule for gradient computation.

    Args:
        circuit: The quantum circuit template with parameters.
        evaluator: The circuit evaluator.
        params: Dictionary mapping parameter names to torch tensor values.

    Returns:
        torch.Tensor: The evaluation result (differentiable w.r.t. params).

    Example:
        >>> from qiskit.circuit import QuantumCircuit, Parameter
        >>> from qtpu.torch import ExpvalTorchEvaluator
        >>> 
        >>> theta = Parameter('theta')
        >>> qc = QuantumCircuit(1)
        >>> qc.ry(theta, 0)
        >>> qc.measure_all()
        >>> 
        >>> evaluator = ExpvalTorchEvaluator()
        >>> param = torch.tensor(0.5, requires_grad=True)
        >>> result = differentiable_evaluate(qc, evaluator, {'theta': param})
        >>> result.backward()
        >>> print(param.grad)  # Gradient via parameter-shift
    """
    # Sort params for consistent ordering
    param_names = sorted(params.keys())
    param_values = [params[name] for name in param_names]
    
    return ParameterShiftFunction.apply(circuit, evaluator, param_names, *param_values)


class DifferentiableTorchEvaluator:
    """Wrapper that makes any TorchCircuitEvaluator differentiable.
    
    Wraps an evaluator to automatically use parameter-shift rule for gradients.
    
    Example:
        >>> base_evaluator = ExpvalTorchEvaluator()
        >>> diff_evaluator = DifferentiableTorchEvaluator(base_evaluator)
        >>> 
        >>> # Now evaluations are differentiable
        >>> result = diff_evaluator.evaluate(circuit, params)
        >>> result.backward()  # Works!
    """

    def __init__(self, base_evaluator: TorchCircuitEvaluator) -> None:
        """Initialize with a base evaluator.

        Args:
            base_evaluator: The underlying circuit evaluator.
        """
        self._base = base_evaluator

    def evaluate(
        self,
        circuit: QuantumCircuit,
        params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Evaluate with automatic differentiation support.

        Args:
            circuit: The quantum circuit template.
            params: Parameter values as torch tensors.

        Returns:
            torch.Tensor: Differentiable evaluation result.
        """
        return differentiable_evaluate(circuit, self._base, params)

    def evaluate_batch(
        self,
        circuits: list[QuantumCircuit],
        params: dict[str, torch.Tensor],
    ) -> list[torch.Tensor]:
        """Evaluate batch of circuits with differentiation support.

        Args:
            circuits: List of quantum circuits.
            params: Parameter values (shared across circuits).

        Returns:
            list[torch.Tensor]: List of differentiable results.
        """
        return [self.evaluate(circuit, params) for circuit in circuits]
