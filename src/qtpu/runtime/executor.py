"""Main HEinsum runtime executor."""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, Callable

import cotengra as ctg
import numpy as np
import torch
from torch.autograd import Function

from qtpu.runtime.backends import (
    QuantumBackend,
    SimulatorBackend,
    FakeQPUBackend,
    FakeQPUCudaQBackend,
    CudaQBackend,
)
from qtpu.runtime.device import get_device, Device
from qtpu.runtime.timing import TimingBreakdown

if TYPE_CHECKING:
    from qtpu.core import HEinsum, QuantumTensor


class _QuantumTensorFunction(Function):
    """PyTorch autograd function for differentiable quantum tensor evaluation.
    
    Uses the parameter-shift rule for gradient computation.
    """
    
    @staticmethod
    def forward(
        ctx,
        qtensor: "QuantumTensor",
        backend: QuantumBackend,
        dtype: torch.dtype,
        device: torch.device,
        param_names: tuple[str, ...],
        *param_values: torch.Tensor,
    ) -> torch.Tensor:
        ctx.qtensor = qtensor
        ctx.backend = backend
        ctx.dtype = dtype
        ctx.device = device
        ctx.param_names = param_names
        ctx.save_for_backward(*param_values)
        
        params = {
            name: float(val.detach().cpu())
            for name, val in zip(param_names, param_values)
        }
        
        result, _, _ = backend.evaluate(qtensor, params, dtype, device)
        return result.clone()
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Parameter-shift gradient: ∂f/∂θ = (1/2)[f(θ + π/2) - f(θ - π/2)]"""
        qtensor = ctx.qtensor
        backend = ctx.backend
        dtype = ctx.dtype
        device = ctx.device
        param_names = ctx.param_names
        param_values = ctx.saved_tensors
        
        shift = np.pi / 2
        grads = []
        
        base_params = {
            name: float(val.detach().cpu())
            for name, val in zip(param_names, param_values)
        }
        
        for i, name in enumerate(param_names):
            params_plus = base_params.copy()
            params_plus[name] = base_params[name] + shift
            result_plus, _, _ = backend.evaluate(qtensor, params_plus, dtype, device)
            
            params_minus = base_params.copy()
            params_minus[name] = base_params[name] - shift
            result_minus, _, _ = backend.evaluate(qtensor, params_minus, dtype, device)
            
            param_grad = 0.5 * (result_plus - result_minus)
            grad = (grad_output * param_grad).sum()
            grads.append(grad.reshape(param_values[i].shape))
        
        # None for: qtensor, backend, dtype, device, param_names
        return (None, None, None, None, None) + tuple(grads)


class HEinsumRuntime:
    """High-performance runtime for hybrid einsum contraction.
    
    Features:
    - Multiple quantum backends (simulator, fake_qpu)
    - GPU acceleration for classical contraction
    - JIT compilation with torch.compile
    - Detailed timing breakdowns
    - Quantum tensor caching for training loops
    
    Args:
        heinsum: The HEinsum specification to execute.
        backend: Quantum backend - "simulator", "fake_qpu", or QuantumBackend instance.
        device: Device for classical computation (None = auto-select).
        dtype: Data type for tensors.
    
    Example:
        >>> runtime = HEinsumRuntime(heinsum, backend="fake_qpu", device="cuda")
        >>> runtime.prepare(jit=True)
        >>> 
        >>> # Single execution with timing
        >>> result, timing = runtime.execute(input_tensors=[x])
        >>> 
        >>> # Training loop with caching
        >>> runtime.cache_quantum()
        >>> for epoch in range(100):
        ...     result, timing = runtime.execute(input_tensors=[x])
    """
    
    def __init__(
        self,
        heinsum: "HEinsum",
        backend: str | QuantumBackend = "simulator",
        device: str | Device | torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ):
        self.heinsum = heinsum
        self.dtype = dtype
        self.device = get_device(device)
        
        # Initialize quantum backend
        if isinstance(backend, str):
            self._backend = self._create_backend(backend)
        else:
            self._backend = backend
        
        # Compilation state
        self._tree: ctg.ContractionTree | None = None
        self._contract_fn: Callable | None = None
        self._prepared = False
        
        # Caching
        self._quantum_cache: dict[int, torch.Tensor] = {}
        self._cache_enabled = False
    
    def _create_backend(self, name: str) -> QuantumBackend:
        """Create a quantum backend by name."""
        if name == "simulator":
            return SimulatorBackend()
        elif name == "fake_qpu":
            return FakeQPUBackend()
        elif name == "cudaq" or name.startswith("cudaq-"):
            # Extract target from name like "cudaq-nvidia" or use default
            if name == "cudaq":
                target = "qpp-cpu"
            else:
                target = name.split("-", 1)[1]
            return CudaQBackend(target=target)
        elif name == "fake_qpu_cudaq" or name.startswith("fake_qpu_cudaq-"):
            # Hybrid backend: CudaQ simulation + Fake QPU timing
            # Format: "fake_qpu_cudaq" or "fake_qpu_cudaq-<target>"
            if name == "fake_qpu_cudaq":
                target = "qpp-cpu"
            else:
                target = name.split("-", 1)[1]
            return FakeQPUCudaQBackend(target=target)
        else:
            raise ValueError(
                f"Unknown backend: {name}. "
                f"Use 'simulator', 'fake_qpu', 'cudaq', 'cudaq-<target>', "
                f"'fake_qpu_cudaq', or 'fake_qpu_cudaq-<target>'."
            )
    
    # -------------------------------------------------------------------------
    # Preparation
    # -------------------------------------------------------------------------
    
    def prepare(
        self,
        optimize: bool = True,
        jit: bool = False,
        slicing_opts: dict | None = None,
        opt_kwargs: dict | None = None,
    ) -> "HEinsumRuntime":
        """Prepare the runtime for execution.
        
        This optimizes the contraction path, optionally JIT-compiles, and
        prepares the quantum backend (e.g., compiles circuits for CudaQ).
        
        Args:
            optimize: Whether to optimize contraction order with cotengra.
            jit: Whether to JIT-compile with torch.compile.
            slicing_opts: Slicing options for large contractions.
                E.g., {"target_size": 2**28} for GPU memory limits.
            opt_kwargs: Additional kwargs for HyperOptimizer.
            
        Returns:
            self for chaining.
        """
        total_start = perf_counter()
        
        # Track preprocessing timing
        self._prep_timing = TimingBreakdown(
            device=str(self.device),
            backend=self._backend.name if hasattr(self._backend, 'name') else str(type(self._backend).__name__),
        )
        
        inputs, output = ctg.utils.eq_to_inputs_output(self.heinsum.einsum_expr)
        
        # Optimization phase
        opt_start = perf_counter()
        if optimize and len(inputs) > 1:
            opt_kwargs = opt_kwargs or {}
            opt_kwargs.setdefault("on_trial_error", "ignore")
            if slicing_opts:
                opt_kwargs["slicing_reconf_opts"] = slicing_opts
            
            opt = ctg.HyperOptimizer(**opt_kwargs)
            try:
                self._tree = opt.search(inputs, output, self.heinsum.size_dict)
            except (KeyError, ValueError):
                self._tree = None
        self._prep_timing.optimization_time = perf_counter() - opt_start
        
        # Build contraction function
        if self._tree is not None:
            if jit:
                tree = self._tree
                self._contract_fn = torch.compile(
                    lambda arrays: tree.contract_core(arrays, backend="torch")
                )
            else:
                self._contract_fn = lambda arrays: self._tree.contract_core(
                    arrays, backend="torch"
                )
        else:
            expr = self.heinsum.einsum_expr
            self._contract_fn = lambda arrays: torch.einsum(expr, *arrays)
        
        # Prepare quantum backend (compiles circuits for CudaQ-based backends)
        backend_prep_start = perf_counter()
        compile_time = self._backend.prepare(self.heinsum.quantum_tensors)
        self._prep_timing.circuit_compilation_time = compile_time
        
        self._prep_timing.total_time = perf_counter() - total_start
        self._prepared = True
        return self
    
    @property
    def prep_timing(self) -> TimingBreakdown | None:
        """Get the preprocessing timing from prepare()."""
        return getattr(self, '_prep_timing', None)
        
        self._prepared = True
        return self
    
    def cache_quantum(
        self,
        params: dict[str, float | torch.Tensor] | None = None,
    ) -> "HEinsumRuntime":
        """Pre-compute and cache quantum tensor results.
        
        Call this before training when quantum circuits have fixed parameters.
        
        Args:
            params: Fixed circuit parameters. If None, uses empty dict.
            
        Returns:
            self for chaining.
        """
        params = params or {}
        float_params = {
            k: float(v.detach().cpu()) if isinstance(v, torch.Tensor) else float(v)
            for k, v in params.items()
        }
        
        self._quantum_cache.clear()
        for i, qtensor in enumerate(self.heinsum.quantum_tensors):
            # Check if all params are available
            circuit_param_names = {p.name for p in qtensor.circuit.parameters}
            iswitch_param_names = set(qtensor.inds)
            free_param_names = circuit_param_names - iswitch_param_names
            
            missing = free_param_names - set(float_params.keys())
            if missing:
                continue
            
            result, _, _ = self._backend.evaluate(
                qtensor, float_params, self.dtype, self.device
            )
            self._quantum_cache[i] = result
        
        self._cache_enabled = True
        return self
    
    def clear_cache(self) -> "HEinsumRuntime":
        """Clear the quantum tensor cache."""
        self._quantum_cache.clear()
        self._cache_enabled = False
        return self
    
    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------
    
    def execute(
        self,
        input_tensors: list[torch.Tensor] | None = None,
        circuit_params: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, TimingBreakdown]:
        """Execute the hybrid einsum contraction.
        
        Args:
            input_tensors: Runtime input tensors.
            circuit_params: Circuit parameters (rotation angles, etc.).
            
        Returns:
            Tuple of:
            - result: The contracted result tensor
            - timing: Detailed timing breakdown
        """
        if not self._prepared:
            self.prepare()
        
        input_tensors = input_tensors or []
        circuit_params = circuit_params or {}
        
        timing = TimingBreakdown(
            device=str(self.device),
            backend=self._backend.name,
        )
        
        total_start = perf_counter()
        
        # Evaluate quantum tensors
        quantum_results, q_eval_time, q_qpu_time, n_circuits = self._eval_quantum(
            circuit_params
        )
        timing.quantum_eval_time = q_eval_time
        timing.quantum_estimated_qpu_time = q_qpu_time
        timing.num_circuits = n_circuits
        
        # Prepare classical tensors
        transfer_start = perf_counter()
        classical_tensors = [
            ct.data.to(dtype=self.dtype, device=self.device)
            for ct in self.heinsum.classical_tensors
        ]
        input_tensors = [
            t.to(dtype=self.dtype, device=self.device) for t in input_tensors
        ]
        timing.data_transfer_time = perf_counter() - transfer_start
        
        # Contract
        operands = quantum_results + classical_tensors + input_tensors
        
        contract_start = perf_counter()
        if self._tree is not None and self._tree.nslices > 1:
            result = self._contract_sliced(operands)
        else:
            result = self._contract_fn(operands)
        timing.classical_contraction_time = perf_counter() - contract_start
        
        timing.total_time = perf_counter() - total_start
        
        return result, timing
    
    def _eval_quantum(
        self,
        params: dict[str, torch.Tensor],
    ) -> tuple[list[torch.Tensor], float, float, int]:
        """Evaluate all quantum tensors.
        
        Returns:
            Tuple of (results, eval_time, estimated_qpu_time, num_circuits)
        """
        requires_grad = any(
            isinstance(v, torch.Tensor) and v.requires_grad
            for v in params.values()
        ) if params else False
        
        float_params = {
            k: float(v.detach().cpu()) if isinstance(v, torch.Tensor) else float(v)
            for k, v in params.items()
        }
        
        results = []
        total_eval_time = 0.0
        total_qpu_time = 0.0
        total_circuits = 0
        
        for i, qtensor in enumerate(self.heinsum.quantum_tensors):
            if self._cache_enabled and i in self._quantum_cache:
                results.append(self._quantum_cache[i])
                continue
            
            if requires_grad:
                # Use autograd function
                param_names = tuple(sorted(params.keys()))
                param_values = [params[name] for name in param_names]
                result = _QuantumTensorFunction.apply(
                    qtensor, self._backend, self.dtype, self.device,
                    param_names, *param_values
                )
                results.append(result)
                total_circuits += int(np.prod(qtensor.shape)) if qtensor.shape else 1
            else:
                # Backend.evaluate returns (result, eval_time, qpu_time)
                result, eval_time, qpu_time = self._backend.evaluate(
                    qtensor, float_params, self.dtype, self.device
                )
                results.append(result)
                total_eval_time += eval_time
                total_qpu_time += qpu_time
                total_circuits += int(np.prod(qtensor.shape)) if qtensor.shape else 1
        
        return results, total_eval_time, total_qpu_time, total_circuits
    
    def _contract_sliced(self, operands: list[torch.Tensor]) -> torch.Tensor:
        """Contract with slicing for large tensor networks."""
        assert self._tree is not None
        
        slices = []
        for i in range(self._tree.nslices):
            sliced = self._tree.slice_arrays(operands, i)
            result = self._contract_fn(sliced)
            slices.append(result)
        
        return self._tree.gather_slices(slices)
    
    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    
    @property
    def backend(self) -> QuantumBackend:
        """The quantum backend."""
        return self._backend
    
    @property
    def tree(self) -> ctg.ContractionTree | None:
        """The optimized contraction tree."""
        return self._tree
    
    @property
    def prepared(self) -> bool:
        """Whether the runtime has been prepared."""
        return self._prepared
    
    @property
    def num_slices(self) -> int:
        """Number of slices for contraction."""
        return self._tree.nslices if self._tree is not None else 1
    
    @property
    def contraction_cost(self) -> float:
        """Estimated contraction cost (FLOPs)."""
        if self._tree is None:
            return 0.0
        return self._tree.contraction_cost()
    
    @property
    def cache_enabled(self) -> bool:
        """Whether quantum caching is enabled."""
        return self._cache_enabled
    
    @property
    def num_cached(self) -> int:
        """Number of cached quantum tensors."""
        return len(self._quantum_cache)


class HEinsumContractor(HEinsumRuntime):
    """Backward-compatible alias for HEinsumRuntime.
    
    .. deprecated::
        Use HEinsumRuntime instead. This alias will be removed in a future version.
    """
    
    def __init__(
        self,
        heinsum: "HEinsum",
        evaluator=None,  # Ignored, for backward compat
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__(heinsum, backend="simulator", dtype=dtype)
    
    def precompute_quantum(self, circuit_params=None) -> "HEinsumContractor":
        """Backward-compatible alias for cache_quantum."""
        return self.cache_quantum(circuit_params)
    
    def contract(
        self,
        input_tensors: list[torch.Tensor],
        circuit_params: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Backward-compatible contract method (returns only result)."""
        result, _ = self.execute(input_tensors, circuit_params)
        return result
    
    @property
    def compiled(self) -> bool:
        """Backward-compatible alias for prepared."""
        return self._prepared
    
    @property
    def nslices(self) -> int:
        """Backward-compatible alias for num_slices."""
        return self.num_slices
    
    @property
    def quantum_cached(self) -> bool:
        """Whether all quantum tensors are cached."""
        return len(self._quantum_cache) == len(self.heinsum.quantum_tensors)


# Legacy alias
QuantumTensorFunction = _QuantumTensorFunction
