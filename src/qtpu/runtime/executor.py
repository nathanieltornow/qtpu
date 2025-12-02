"""Main HEinsum runtime executor."""

from __future__ import annotations

import itertools
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
    # Approximate Monte Carlo Contraction
    # -------------------------------------------------------------------------
    
    def execute_approximate(
        self,
        num_samples: int,
        input_tensors: list[torch.Tensor] | None = None,
        circuit_params: dict[str, torch.Tensor] | None = None,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, TimingBreakdown, dict]:
        """Execute approximate tensor network contraction via Monte Carlo sampling.
        
        Instead of computing all tensor elements (exponential in the number of
        inner indices), this samples from the index space of inner (contracted)
        indices and performs a Monte Carlo estimation of the contraction.
        
        The method:
        1. Identifies inner indices (those being contracted, not in output)
        2. Samples random index assignments for inner indices
        3. For each sample, evaluates quantum tensors only at those indices
        4. Computes the contraction contribution from each sample
        5. Returns the Monte Carlo average (scaled by volume)
        
        This is useful for large tensor networks where exact contraction is
        infeasible. The error decreases as O(1/sqrt(num_samples)).
        
        Args:
            num_samples: Number of Monte Carlo samples to take.
            input_tensors: Runtime input tensors (must have no inner indices).
            circuit_params: Circuit parameters (rotation angles, etc.).
            seed: Random seed for reproducibility.
            
        Returns:
            Tuple of:
            - result: The approximate contracted result tensor
            - timing: Detailed timing breakdown
            - stats: Dictionary with Monte Carlo statistics:
                - "inner_indices": List of inner index names
                - "inner_volume": Total number of inner index combinations
                - "num_samples": Number of samples used
                - "samples": List of sampled index assignments
                - "values": List of (sample_contribution, weight) for each sample
        
        Example:
            >>> runtime = HEinsumRuntime(heinsum, backend="cudaq")
            >>> runtime.prepare()
            >>> # Approximate contraction with 1000 samples
            >>> result, timing, stats = runtime.execute_approximate(
            ...     num_samples=1000,
            ...     circuit_params={"theta": 0.5},
            ... )
            >>> print(f"Inner volume: {stats['inner_volume']}, Samples: {stats['num_samples']}")
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
        
        # Convert params to float
        float_params = {
            k: float(v.detach().cpu()) if isinstance(v, torch.Tensor) else float(v)
            for k, v in circuit_params.items()
        }
        
        # Parse einsum expression to find inner indices
        inner_info = self._get_inner_index_info()
        inner_indices = inner_info["inner_indices"]
        inner_sizes = inner_info["inner_sizes"]
        inner_volume = inner_info["inner_volume"]
        tensor_inner_inds = inner_info["tensor_inner_inds"]
        
        stats = {
            "inner_indices": list(inner_indices),
            "inner_volume": inner_volume,
            "num_samples": num_samples,
            "samples": [],
            "values": [],
        }
        
        # If no inner indices, fall back to exact contraction
        if not inner_indices:
            result, timing = self.execute(input_tensors, circuit_params)
            return result, timing, stats
        
        # Sample inner index assignments
        if seed is not None:
            np.random.seed(seed)
        
        sampling_start = perf_counter()
        
        # Generate random samples from the inner index space
        inner_ind_list = list(inner_indices)
        inner_dims = [inner_sizes[ind] for ind in inner_ind_list]
        
        # Sample indices
        sampled_inner_assignments = []
        for _ in range(num_samples):
            assignment = tuple(np.random.randint(0, dim) for dim in inner_dims)
            sampled_inner_assignments.append(assignment)
        
        stats["samples"] = sampled_inner_assignments
        timing.data_transfer_time = perf_counter() - sampling_start
        
        # For each quantum tensor, determine which indices need to be sampled
        quantum_eval_start = perf_counter()
        
        # Build index mappings for each tensor
        qtensor_samples_map = self._build_tensor_sample_indices(
            self.heinsum.quantum_tensors,
            tensor_inner_inds,
            inner_ind_list,
            sampled_inner_assignments,
        )
        
        # Sample quantum tensors
        quantum_sample_results = []
        total_circuits = 0
        for i, qtensor in enumerate(self.heinsum.quantum_tensors):
            if i in qtensor_samples_map and qtensor_samples_map[i]:
                indices_to_sample = qtensor_samples_map[i]
                samples = self._backend.sample(qtensor, indices_to_sample, float_params)
                # Convert to dict for fast lookup
                sample_dict = {idx: val for idx, val in samples}
                quantum_sample_results.append(sample_dict)
                total_circuits += len(indices_to_sample)
            else:
                # No inner indices for this tensor - evaluate fully
                result, _, _ = self._backend.evaluate(
                    qtensor, float_params, self.dtype, self.device
                )
                quantum_sample_results.append(result.numpy())
                total_circuits += int(np.prod(qtensor.shape)) if qtensor.shape else 1
        
        timing.quantum_eval_time = perf_counter() - quantum_eval_start
        timing.num_circuits = total_circuits
        
        # Prepare classical tensors (full tensors)
        transfer_start = perf_counter()
        classical_tensors = [
            ct.data.to(dtype=self.dtype, device=self.device).numpy()
            for ct in self.heinsum.classical_tensors
        ]
        input_arrays = [
            t.to(dtype=self.dtype, device=self.device).numpy() for t in input_tensors
        ]
        timing.data_transfer_time += perf_counter() - transfer_start
        
        # Perform Monte Carlo contraction
        contract_start = perf_counter()
        result = self._monte_carlo_contract(
            sampled_inner_assignments,
            inner_ind_list,
            inner_volume,
            quantum_sample_results,
            classical_tensors,
            input_arrays,
            tensor_inner_inds,
            stats,
        )
        timing.classical_contraction_time = perf_counter() - contract_start
        
        # Convert result to torch tensor
        result = torch.tensor(result, dtype=self.dtype, device=self.device)
        
        timing.total_time = perf_counter() - total_start
        
        return result, timing, stats
    
    def _get_inner_index_info(self) -> dict:
        """Extract information about inner (contracted) indices.
        
        Returns:
            Dictionary with:
            - inner_indices: set of inner index names (char form)
            - inner_sizes: dict mapping inner index char to size
            - inner_volume: total product of inner index sizes
            - tensor_inner_inds: list of (tensor_idx, inner_ind_positions) tuples
        """
        inputs, output = ctg.utils.eq_to_inputs_output(self.heinsum.einsum_expr)
        
        # All indices appearing in inputs
        all_inds = set()
        for inp in inputs:
            all_inds.update(inp)
        
        # Inner indices = all - output
        output_inds = set(output)
        inner_indices = all_inds - output_inds
        
        # Get sizes for inner indices
        inner_sizes = {ind: self.heinsum.size_dict[ind] for ind in inner_indices}
        inner_volume = 1
        for size in inner_sizes.values():
            inner_volume *= size
        
        # For each tensor, find which positions correspond to inner indices
        all_tensors = (
            list(self.heinsum.quantum_tensors)
            + list(self.heinsum.classical_tensors)
            + list(self.heinsum.input_tensors)
        )
        
        tensor_inner_inds = []
        for i, inp in enumerate(inputs):
            positions = {}
            for pos, ind in enumerate(inp):
                if ind in inner_indices:
                    positions[ind] = pos
            tensor_inner_inds.append(positions)
        
        return {
            "inner_indices": inner_indices,
            "inner_sizes": inner_sizes,
            "inner_volume": inner_volume,
            "tensor_inner_inds": tensor_inner_inds,
        }
    
    def _build_tensor_sample_indices(
        self,
        tensors: list,
        tensor_inner_inds: list[dict],
        inner_ind_list: list[str],
        sampled_assignments: list[tuple],
    ) -> dict[int, list[tuple]]:
        """Build the set of tensor indices to sample for each quantum tensor.
        
        For each quantum tensor with inner indices, we need to know which
        full tensor indices to sample. This maps the inner index assignments
        to the appropriate tensor indices.
        
        Args:
            tensors: List of tensors (quantum tensors).
            tensor_inner_inds: For each tensor position in einsum, dict mapping
                inner index char to position in tensor.
            inner_ind_list: Ordered list of inner index names.
            sampled_assignments: List of sampled inner index assignments.
            
        Returns:
            Dict mapping tensor index to list of tensor indices to sample.
        """
        result = {}
        n_quantum = len(tensors)
        
        for tensor_idx in range(n_quantum):
            inner_pos_map = tensor_inner_inds[tensor_idx]
            
            if not inner_pos_map:
                # No inner indices for this tensor
                continue
            
            tensor = tensors[tensor_idx]
            tensor_shape = tensor.shape
            n_dims = len(tensor_shape)
            
            # For each sample, build the full tensor index
            # We need to enumerate over outer indices of this tensor
            indices_to_sample = set()
            
            for assignment in sampled_assignments:
                # Build assignment dict: inner_ind -> value
                assign_dict = {
                    inner_ind_list[i]: assignment[i]
                    for i in range(len(inner_ind_list))
                }
                
                # Get the tensor's einsum indices (from parsed expression)
                inputs, _ = ctg.utils.eq_to_inputs_output(self.heinsum.einsum_expr)
                tensor_inds = inputs[tensor_idx]
                
                # Build the partial index (just inner indices known)
                # For outer indices of this tensor, we need all combinations
                outer_positions = []
                outer_sizes = []
                inner_values = [None] * n_dims
                
                for pos, ind in enumerate(tensor_inds):
                    if ind in assign_dict:
                        inner_values[pos] = assign_dict[ind]
                    else:
                        outer_positions.append(pos)
                        outer_sizes.append(tensor_shape[pos])
                
                # Generate all combinations of outer indices
                if outer_positions:
                    for outer_vals in itertools.product(*[range(s) for s in outer_sizes]):
                        full_idx = list(inner_values)
                        for i, pos in enumerate(outer_positions):
                            full_idx[pos] = outer_vals[i]
                        indices_to_sample.add(tuple(full_idx))
                else:
                    indices_to_sample.add(tuple(inner_values))
            
            result[tensor_idx] = list(indices_to_sample)
        
        return result
    
    def _monte_carlo_contract(
        self,
        sampled_assignments: list[tuple],
        inner_ind_list: list[str],
        inner_volume: int,
        quantum_results: list,
        classical_arrays: list[np.ndarray],
        input_arrays: list[np.ndarray],
        tensor_inner_inds: list[dict],
        stats: dict,
    ) -> np.ndarray:
        """Perform Monte Carlo contraction over sampled inner indices.
        
        For each sampled inner index assignment:
        1. Extract the slice of each tensor at that inner index
        2. Contract the remaining outer indices
        3. Accumulate the result
        4. Scale by (inner_volume / num_samples)
        
        Args:
            sampled_assignments: List of sampled (inner_idx_1, inner_idx_2, ...) tuples.
            inner_ind_list: Ordered list of inner index chars.
            inner_volume: Total number of inner index combinations.
            quantum_results: List of either dict (sampled) or ndarray (full).
            classical_arrays: List of classical numpy arrays.
            input_arrays: List of input numpy arrays.
            tensor_inner_inds: For each tensor position, dict mapping inner ind to pos.
            stats: Stats dict to update with sample values.
            
        Returns:
            Approximate contraction result as numpy array.
        """
        inputs, output = ctg.utils.eq_to_inputs_output(self.heinsum.einsum_expr)
        n_quantum = len(quantum_results)
        n_classical = len(classical_arrays)
        n_input = len(input_arrays)
        
        # Convert inner_ind_list to set for fast lookup
        inner_ind_set = set(inner_ind_list)
        
        # Build the reduced einsum expression (with inner indices removed)
        reduced_inputs = []
        for i, inp in enumerate(inputs):
            reduced = "".join(c for c in inp if c not in inner_ind_set)
            reduced_inputs.append(reduced)
        # output is a tuple of chars, convert to string
        output_str = "".join(output)
        reduced_expr = ",".join(reduced_inputs) + "->" + output_str
        
        # Determine output shape
        output_shape = tuple(
            self.heinsum.size_dict[c] for c in output
        ) if output else ()
        
        # Accumulator for Monte Carlo sum
        result_sum = np.zeros(output_shape, dtype=np.float64)
        
        for assignment in sampled_assignments:
            # Build assignment dict: inner_ind -> value
            assign_dict = {
                inner_ind_list[i]: assignment[i]
                for i in range(len(inner_ind_list))
            }
            
            # Extract slices from all tensors
            sliced_operands = []
            
            # Process each tensor
            for tensor_idx in range(n_quantum + n_classical + n_input):
                inner_pos_map = tensor_inner_inds[tensor_idx]
                tensor_inds = inputs[tensor_idx]
                
                if tensor_idx < n_quantum:
                    # Quantum tensor
                    q_result = quantum_results[tensor_idx]
                    if isinstance(q_result, dict):
                        # Sampled results - need to build sliced tensor
                        # Determine outer shape
                        outer_inds = [c for c in tensor_inds if c not in inner_ind_set]
                        outer_shape = tuple(
                            self.heinsum.size_dict[c] for c in outer_inds
                        ) if outer_inds else ()
                        
                        if outer_shape:
                            sliced = np.zeros(outer_shape, dtype=np.float64)
                            # Fill in from sampled results
                            for full_idx, val in q_result.items():
                                # Extract outer portion of full_idx
                                outer_idx = tuple(
                                    full_idx[pos] for pos, c in enumerate(tensor_inds)
                                    if c not in inner_ind_set
                                )
                                # Check if this sample matches our inner assignment
                                matches = True
                                for ind, pos in inner_pos_map.items():
                                    if full_idx[pos] != assign_dict[ind]:
                                        matches = False
                                        break
                                if matches:
                                    sliced[outer_idx] = val
                            sliced_operands.append(sliced)
                        else:
                            # Scalar output - find matching sample
                            val = 0.0
                            full_idx = tuple(
                                assign_dict.get(c, 0) for c in tensor_inds
                            )
                            if full_idx in q_result:
                                val = q_result[full_idx]
                            sliced_operands.append(np.array(val))
                    else:
                        # Full tensor result
                        sliced = self._slice_tensor(
                            q_result, tensor_inds, inner_pos_map, assign_dict
                        )
                        sliced_operands.append(sliced)
                        
                elif tensor_idx < n_quantum + n_classical:
                    # Classical tensor
                    c_idx = tensor_idx - n_quantum
                    sliced = self._slice_tensor(
                        classical_arrays[c_idx], tensor_inds, inner_pos_map, assign_dict
                    )
                    sliced_operands.append(sliced)
                    
                else:
                    # Input tensor
                    i_idx = tensor_idx - n_quantum - n_classical
                    sliced = self._slice_tensor(
                        input_arrays[i_idx], tensor_inds, inner_pos_map, assign_dict
                    )
                    sliced_operands.append(sliced)
            
            # Contract the sliced operands
            if len(sliced_operands) == 1:
                contribution = sliced_operands[0]
            else:
                contribution = np.einsum(reduced_expr, *sliced_operands)
            
            # Accumulate
            result_sum += contribution
            stats["values"].append((float(np.sum(contribution)), 1.0))
        
        # Scale by volume / num_samples (Monte Carlo estimator)
        num_samples = len(sampled_assignments)
        result = result_sum * (inner_volume / num_samples)
        
        return result
    
    def _slice_tensor(
        self,
        tensor: np.ndarray,
        tensor_inds: tuple[str, ...],
        inner_pos_map: dict[str, int],
        assign_dict: dict[str, int],
    ) -> np.ndarray:
        """Extract a slice of a tensor by fixing inner indices.
        
        Args:
            tensor: The tensor to slice.
            tensor_inds: Index characters for each dimension.
            inner_pos_map: Dict mapping inner index char to position.
            assign_dict: Dict mapping inner index char to assigned value.
            
        Returns:
            Sliced tensor with inner dimensions removed.
        """
        if not inner_pos_map:
            return tensor
        
        # Build slice tuple
        slices = []
        for pos, ind in enumerate(tensor_inds):
            if ind in assign_dict:
                slices.append(assign_dict[ind])
            else:
                slices.append(slice(None))
        
        return tensor[tuple(slices)]

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
