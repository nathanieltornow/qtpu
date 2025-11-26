"""Differentiable HEinsum contraction for PyTorch."""

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
    """Reusable contractor for HEinsum with JIT compilation support."""

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
        
        # Quantum tensor cache for fixed (non-parameterized) circuits
        self._quantum_cache: dict[int, torch.Tensor] = {}
        self._cache_enabled = True

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
    
    def precompute_quantum(
        self,
        circuit_params: dict[str, torch.Tensor] | None = None,
    ) -> "HEinsumContractor":
        """Pre-compute and cache quantum tensor evaluations.
        
        Call this once before training when quantum circuits have no trainable
        parameters (or only some do). Only caches tensors whose parameters are
        all provided in circuit_params. Tensors with missing/trainable params
        will be evaluated on each contract() call.
        
        Args:
            circuit_params: Fixed circuit parameters (should not require gradients).
            
        Returns:
            self for chaining.
            
        Example:
            >>> contractor = HEinsumContractor(heinsum)
            >>> contractor.prepare()
            >>> contractor.precompute_quantum()  # Cache quantum results
            >>> 
            >>> # Now training is fast - no quantum re-evaluation
            >>> for epoch in range(100):
            ...     output = contractor.contract(input_tensors=[x], circuit_params={})
        """
        if circuit_params is None:
            circuit_params = {}
            
        # Convert params to floats
        float_params = {
            k: float(v.detach().cpu()) if isinstance(v, torch.Tensor) else float(v)
            for k, v in circuit_params.items()
        }
        
        # Evaluate and cache quantum tensors that have all params available
        self._quantum_cache.clear()
        for i, qtensor in enumerate(self.heinsum.quantum_tensors):
            # Get free parameters (circuit params that aren't ISwitch indices)
            circuit_param_names = {p.name for p in qtensor.circuit.parameters}
            iswitch_param_names = set(qtensor.inds)
            free_param_names = circuit_param_names - iswitch_param_names
            
            # Check if all free params are provided
            missing_params = free_param_names - set(float_params.keys())
            
            if missing_params:
                # Skip caching - this tensor has trainable/missing parameters
                continue
                
            result = self.evaluator.evaluate_with_params(qtensor, float_params)
            self._quantum_cache[i] = result.to(self.dtype)
            
        return self
    
    def clear_cache(self) -> "HEinsumContractor":
        """Clear the quantum tensor cache.
        
        Call this if you need to re-evaluate quantum tensors with different
        parameters.
        
        Returns:
            self for chaining.
        """
        self._quantum_cache.clear()
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
    def quantum_cached(self) -> bool:
        """Whether all quantum tensors are cached."""
        return len(self._quantum_cache) == len(self.heinsum.quantum_tensors)
    
    @property
    def num_cached(self) -> int:
        """Number of quantum tensors that are cached."""
        return len(self._quantum_cache)

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
                If empty and quantum tensors are cached, uses cached values.

        Returns:
            torch.Tensor: The contracted result. Differentiable w.r.t.:
                - circuit_params (via parameter-shift rule)
                - input_tensors (via torch autograd)
                - classical tensor data (via torch autograd)
        """
        if circuit_params is None:
            circuit_params = {}

        # Get quantum tensor values (from cache or by evaluation)
        quantum_results = self._get_quantum_tensors(circuit_params)

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
    
    def _get_quantum_tensors(
        self,
        circuit_params: dict[str, torch.Tensor],
    ) -> list[torch.Tensor]:
        """Get quantum tensor values, using cache where available."""
        
        # Check if any parameters require gradients
        requires_grad = (
            any(isinstance(v, torch.Tensor) and v.requires_grad for v in circuit_params.values())
            if circuit_params
            else False
        )
        
        # Evaluate quantum tensors, using cache where available
        quantum_results = []
        for i, qtensor in enumerate(self.heinsum.quantum_tensors):
            # Use cache if this tensor is cached
            if self._cache_enabled and i in self._quantum_cache:
                result = self._quantum_cache[i]
            else:
                # Evaluate this tensor (it has trainable params or wasn't cached)
                result = _evaluate_quantum_tensor(qtensor, self.evaluator, circuit_params, requires_grad)
                result = result.to(self.dtype)
                    
            quantum_results.append(result)
            
        return quantum_results

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
    requires_grad: bool | None = None,
) -> torch.Tensor:
    """Evaluate a quantum tensor with optional differentiable parameters.
    
    Args:
        qtensor: The quantum tensor to evaluate.
        evaluator: Evaluator for quantum circuits.
        params: Dict of circuit parameters.
        requires_grad: Whether gradients are needed. If None, auto-detect.
    """

    # Auto-detect if any parameters require gradients
    if requires_grad is None:
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
