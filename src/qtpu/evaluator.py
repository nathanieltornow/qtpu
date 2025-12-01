"""Differentiable quantum tensor evaluation for PyTorch."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.autograd import Function

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit
    from qiskit.primitives import BaseEstimatorV2
    from qtpu.core.tensor import QuantumTensor


class QuantumTensorFunction(Function):
    """PyTorch autograd function for differentiable quantum tensor evaluation.

    This function:
    - Forward: Evaluates all configurations of a quantum tensor with given parameters
    - Backward: Computes gradients using the parameter-shift rule

    Assumes:
    - ISwitch operations define tensor indices (not trainable)
    - Other circuit parameters (rotation angles, etc.) are trainable
    """

    @staticmethod
    def forward(
        ctx,
        qtensor: QuantumTensor,
        evaluator: "QuantumTensorEvaluator",
        param_names: tuple[str, ...],
        *param_values: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate quantum tensor with current parameter values.

        Args:
            ctx: Autograd context.
            qtensor: The quantum tensor to evaluate.
            evaluator: Evaluator for quantum circuits.
            param_names: Tuple of parameter names (must match circuit parameters).
            *param_values: Parameter values as torch tensors.

        Returns:
            torch.Tensor: Evaluated tensor with shape qtensor.shape.
        """
        ctx.qtensor = qtensor
        ctx.evaluator = evaluator
        ctx.param_names = param_names
        ctx.save_for_backward(*param_values)

        # Build parameter dict
        params = {
            name: float(val.detach().cpu())
            for name, val in zip(param_names, param_values)
        }

        # Evaluate
        result = evaluator.evaluate_with_params(qtensor, params)
        return result.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Compute gradients using parameter-shift rule.

        For rotation gates: ∂f/∂θ = (1/2)[f(θ + π/2) - f(θ - π/2)]
        """
        qtensor = ctx.qtensor
        evaluator = ctx.evaluator
        param_names = ctx.param_names
        param_values = ctx.saved_tensors

        shift = np.pi / 2
        grads = []

        # Base parameters
        base_params = {
            name: float(val.detach().cpu())
            for name, val in zip(param_names, param_values)
        }

        for i, name in enumerate(param_names):
            # Shift up
            params_plus = base_params.copy()
            params_plus[name] = base_params[name] + shift
            result_plus = evaluator.evaluate_with_params(qtensor, params_plus)

            # Shift down
            params_minus = base_params.copy()
            params_minus[name] = base_params[name] - shift
            result_minus = evaluator.evaluate_with_params(qtensor, params_minus)

            # Parameter-shift gradient
            param_grad = 0.5 * (result_plus - result_minus)

            # Chain rule: sum over output dimensions
            grad = (grad_output * param_grad).sum()
            grads.append(grad.reshape(param_values[i].shape))

        # Return: None for qtensor, evaluator, param_names, then gradients for each param
        return (None, None, None) + tuple(grads)


class QuantumTensorEvaluator:
    """Simple evaluator for quantum tensors that returns PyTorch tensors.

    This evaluator:
    1. Takes a quantum tensor and parameter assignments
    2. Evaluates all ISwitch configurations
    3. Returns results as a PyTorch tensor
    """

    def __init__(
        self,
        estimator: "BaseEstimatorV2 | None" = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize the evaluator.

        Args:
            estimator: Qiskit estimator primitive. If None, uses Aer EstimatorV2.
            device: PyTorch device for output tensors.
            dtype: PyTorch dtype for output tensors.
        """
        if estimator is None:
            from qiskit_aer.primitives import EstimatorV2

            estimator = EstimatorV2()

        self.estimator = estimator
        self.device = device
        self.dtype = dtype

    def evaluate_with_params(
        self,
        qtensor: QuantumTensor,
        params: dict[str, float],
    ) -> torch.Tensor:
        """Evaluate quantum tensor with specific parameter values.

        Args:
            qtensor: The quantum tensor to evaluate.
            params: Dict mapping parameter names to float values.

        Returns:
            torch.Tensor: Evaluated tensor with shape qtensor.shape.
        """
        from qtpu.transforms import decompose_qpd_measures, remove_operations_by_name

        # Get all circuit configurations (indexed by ISwitch)
        circuits = qtensor.flat()

        # Process each circuit
        bound_circuits = []
        observables = []

        for circuit in circuits:
            # Decompose to expand ISwitch
            circuit = circuit.decompose()

            # Bind trainable parameters
            if circuit.parameters and params:
                # Only bind parameters that exist in this circuit
                circuit_param_names = {p.name for p in circuit.parameters}
                params_to_bind = {
                    k: v for k, v in params.items() if k in circuit_param_names
                }
                if params_to_bind:
                    circuit = circuit.assign_parameters(params_to_bind)

            # Handle QPD measures if present
            circuit = decompose_qpd_measures(circuit, defer=True, inplace=True)
            circuit = circuit.decompose()

            # Get observable
            obs = self._get_z_observable(circuit)

            # Remove measurements for estimator
            circuit = circuit.remove_final_measurements(inplace=False)
            remove_operations_by_name(circuit, {"reset"})

            bound_circuits.append(circuit)
            observables.append(obs)

        # Run estimator
        jobs = list(zip(bound_circuits, observables))
        results = self.estimator.run(jobs).result()

        # Extract expectation values
        expvals = [r.data.evs for r in results]

        # Reshape to tensor shape
        data = np.array(expvals).reshape(qtensor.shape)
        return torch.tensor(data, dtype=self.dtype, device=self.device)

    def _get_z_observable(self, circuit: "QuantumCircuit") -> str:
        """Get Z observable string for measured qubits."""
        measured = set()
        for instr in circuit:
            if instr.operation.name == "measure":
                measured.add(circuit.qubits.index(instr.qubits[0]))

        if not measured:
            return "Z" * circuit.num_qubits

        obs = ["I"] * circuit.num_qubits
        for q in measured:
            obs[q] = "Z"
        return "".join(reversed(obs))
