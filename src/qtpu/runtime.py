"""Differentiable HEinsum contraction for PyTorch.

Provides a simple `contract` function that users can call from their own
PyTorch modules with their own learnable parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import cotengra as ctg
import torch

from qtpu.tensor import QuantumTensor
from qtpu.evaluator import (
    QuantumTensorEvaluator,
    QuantumTensorFunction,
)

if TYPE_CHECKING:
    from qtpu.heinsum import HEinsum


class HEinsumContractor:
    """Reusable contractor for HEinsum with JIT compilation support.

    This class wraps an HEinsum and evaluator, providing a simple `contract()`
    method that can be called with circuit parameters and input tensors.

    Uses cotengra to find optimal contraction paths and supports JIT compilation
    of the core contraction via torch.compile for maximum performance.

    Example:
        >>> from qtpu.heinsum import HEinsum
        >>> from qtpu.torch import HEinsumContractor, QuantumTensorEvaluator
        >>>
        >>> # Create and prepare contractor
        >>> contractor = HEinsumContractor(heinsum)
        >>> contractor.prepare(jit=True)  # Optimize tree + JIT compile
        >>>
        >>> # Use in a custom layer
        >>> class MyLayer(nn.Module):
        ...     def __init__(self, heinsum):
        ...         super().__init__()
        ...         self.contractor = HEinsumContractor(heinsum)
        ...         self.contractor.prepare(jit=True)
        ...         self.theta = nn.Parameter(torch.tensor(0.5))
        ...
        ...     def forward(self, x):
        ...         return self.contractor.contract(
        ...             input_tensors=[x],
        ...             circuit_params={"theta": self.theta},
        ...         )
    """

    def __init__(
        self,
        heinsum: HEinsum,
        evaluator: QuantumTensorEvaluator | None = None,
        dtype: torch.dtype = torch.float64,
    ):
        """Initialize the contractor.

        Args:
            heinsum: The HEinsum specification.
            evaluator: Evaluator for quantum circuits. If None, creates default.
            dtype: PyTorch dtype for computations.
        """
        self.heinsum = heinsum
        self.evaluator = evaluator or QuantumTensorEvaluator(dtype=dtype)
        self.dtype = dtype

        # Compilation state
        self._tree: ctg.ContractionTree | None = None
        self._contract_core_jit: Callable | None = None
        self._compiled = False

    def prepare(
        self,
        optimize: bool = True,
        jit: bool = False,
        slicing_reconf_opts: dict | None = None,
        opt_kwargs: dict | None = None,
    ) -> "HEinsumContractor":
        """Pre-compile the contraction expression for faster execution.

        This method:
        1. Finds an optimized contraction tree using cotengra
        2. Optionally JIT-compiles tree.contract_core with torch.compile

        Args:
            optimize: Whether to optimize the contraction tree.
            jit: Whether to JIT-compile with torch.compile.
            slicing_reconf_opts: Options for slicing large contractions.
                E.g., {"target_size": 2**28} to fit on GPU.
            opt_kwargs: Additional kwargs for HyperOptimizer.

        Returns:
            self for chaining.

        Example:
            >>> # Basic compilation with tree optimization
            >>> contractor = HEinsumContractor(heinsum).prepare()
            >>>
            >>> # With JIT compilation for maximum performance
            >>> contractor.prepare(jit=True)
            >>>
            >>> # With slicing for large contractions (e.g., GPU memory limits)
            >>> contractor.prepare(slicing_reconf_opts={"target_size": 2**28}, jit=True)
            >>>
            >>> # With custom optimization parameters
            >>> contractor.prepare(opt_kwargs={"max_repeats": 64, "parallel": True})
        """
        inputs, output = ctg.utils.eq_to_inputs_output(self.heinsum.einsum_expr)

        # Only optimize if there are multiple tensors to contract
        if optimize and len(inputs) > 1:
            # Build contraction tree
            opt_kwargs = opt_kwargs or {}
            opt_kwargs.setdefault("on_trial_error", "ignore")
            if slicing_reconf_opts:
                opt_kwargs["slicing_reconf_opts"] = slicing_reconf_opts

            opt = ctg.HyperOptimizer(**opt_kwargs)
            try:
                self._tree = opt.search(inputs, output, self.heinsum.char_size_dict)
            except (KeyError, ValueError):
                # Optimization failed, fall back to einsum
                self._tree = None

        # JIT compile the core contraction
        if jit and self._tree is not None:
            # Create JIT-compiled version of contract_core
            # Similar to: jax.jit(tree.contract_core) in the JAX example
            self._contract_core_jit = torch.compile(
                lambda arrays: self._tree.contract_core(arrays, backend="torch")
            )

        self._compiled = True
        return self

    @property
    def tree(self) -> ctg.ContractionTree | None:
        """The optimized contraction tree, if compiled."""
        return self._tree

    @property
    def compiled(self) -> bool:
        """Whether the contractor has been compiled."""
        return self._compiled

    @property
    def nslices(self) -> int:
        """Number of slices for the contraction (1 if no slicing)."""
        return self._tree.nslices if self._tree is not None else 1

    def contract(
        self,
        input_tensors: list[torch.Tensor],
        circuit_params: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Contract the HEinsum with given parameters and inputs.

        Args:
            input_tensors: Runtime input tensors matching heinsum.input_tensors specs.
            circuit_params: Dict of quantum circuit parameters (rotation angles, etc.).
                If any require gradients, parameter-shift rule is used.

        Returns:
            torch.Tensor: The contracted result. Differentiable w.r.t.:
                - circuit_params (via parameter-shift rule)
                - input_tensors (via torch autograd)
        """
        if circuit_params is None:
            circuit_params = {}

        # Evaluate quantum tensors
        quantum_results = []
        for qtensor in self.heinsum.quantum_tensors:
            result = _evaluate_quantum_tensor(qtensor, self.evaluator, circuit_params)
            quantum_results.append(result.to(self.dtype))

        # Get classical tensors (already torch.Tensor)
        classical_tensors = [
            ct.data.to(self.dtype) for ct in self.heinsum.classical_tensors
        ]

        # Convert input tensors to correct dtype
        input_tensors = [t.to(self.dtype) for t in input_tensors]

        # Assemble operands: quantum, classical, input
        operands = quantum_results + classical_tensors + input_tensors

        # Contract using the appropriate method
        if self._tree is not None:
            if self._tree.nslices > 1:
                # Sliced contraction for large tensor networks
                return self._contract_sliced(operands)
            elif self._contract_core_jit is not None:
                # JIT-compiled core contraction
                return self._contract_core_jit(operands)
            else:
                # Use cotengra's contract_core directly
                return self._tree.contract_core(operands, backend="torch")
        else:
            # Fall back to torch.einsum
            return torch.einsum(self.heinsum.einsum_expr, *operands)

    def _contract_sliced(self, operands: list[torch.Tensor]) -> torch.Tensor:
        """Contract with slicing for large tensor networks.

        For sliced contractions, we iterate over slices and use contract_core
        (optionally JIT-compiled) for each slice, then gather results.

        This pattern follows the JAX example:
            contract_core_jit = jax.jit(tree.contract_core)
            for i in range(tree.nslices):
                sliced = tree.slice_arrays(arrays, i)
                result = contract_core_jit(sliced)
            tree.gather_slices(slices)
        """
        assert self._tree is not None

        slices = []
        for i in range(self._tree.nslices):
            sliced_arrays = self._tree.slice_arrays(operands, i)

            if self._contract_core_jit is not None:
                # Use JIT-compiled contract_core
                result = self._contract_core_jit(sliced_arrays)
            else:
                # Use contract_core directly
                result = self._tree.contract_core(sliced_arrays, backend="torch")

            slices.append(result)

        return self._tree.gather_slices(slices)


def _evaluate_quantum_tensor(
    qtensor: QuantumTensor,
    evaluator: QuantumTensorEvaluator,
    params: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Evaluate a quantum tensor with optional differentiable parameters."""

    # Check if any parameters require gradients
    requires_grad = (
        any(isinstance(v, torch.Tensor) and v.requires_grad for v in params.values())
        if params
        else False
    )

    if not requires_grad:
        # No gradients needed - direct evaluation
        float_params = {
            k: float(v.detach().cpu()) if isinstance(v, torch.Tensor) else float(v)
            for k, v in params.items()
        }
        return evaluator.evaluate_with_params(qtensor, float_params)

    # Use autograd function for differentiable evaluation
    param_names = tuple(sorted(params.keys()))
    param_values = [params[name] for name in param_names]

    return QuantumTensorFunction.apply(
        qtensor,
        evaluator,
        param_names,
        *param_values,
    )
