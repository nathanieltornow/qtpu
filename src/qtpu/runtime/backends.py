"""Quantum tensor evaluation backend using CUDA-Q."""

from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import torch

try:
    import cudaq
except ImportError:
    cudaq = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from qtpu.core import QuantumTensor
    from qtpu.compiler.codegen import CompiledQuantumTensor


class QuantumBackend(ABC):
    """Abstract base class for quantum tensor evaluation backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name for logging."""
        ...

    def prepare(self, qtensors: list["QuantumTensor"]) -> float:
        """Prepare the backend for evaluating the given quantum tensors.
        
        Returns:
            Total preparation/compilation time in seconds.
        """
        return 0.0

    @abstractmethod
    def evaluate(
        self,
        qtensor: "QuantumTensor",
        params: dict[str, float],
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, float, float]:
        """Evaluate a quantum tensor.

        Returns:
            Tuple of (result, eval_time, estimated_qpu_time).
        """
        ...


class CudaQBackend(QuantumBackend):
    """CUDA-Q backend for quantum tensor evaluation.

    Uses NVIDIA CUDA-Q for JIT-compiled quantum circuit simulation.
    Optionally estimates QPU execution time using Qiskit scheduling.

    Args:
        target: CUDA-Q target backend. Options include:
            - "qpp-cpu" (default): CPU simulation with OpenMP
            - "nvidia": GPU simulation (requires NVIDIA GPU)
            - "nvidia-fp64": GPU with double precision
        simulate: If True, run actual quantum simulation.
            If False, skip execution and return random results (for benchmarking).
        estimate_qpu_time: If True, estimate QPU execution time using Qiskit.
        backend_name: Fake backend name for QPU time estimation (e.g., "FakeMarrakesh").
        shots: Number of shots for time estimation.
        optimization_level: Transpilation optimization level (0-3).

    Example:
        >>> # Full simulation
        >>> backend = CudaQBackend(target="nvidia")
        >>> 
        >>> # Benchmarking mode (no simulation, just timing estimation)
        >>> backend = CudaQBackend(simulate=False, estimate_qpu_time=True)
        >>> 
        >>> # Fast mode (no simulation, no QPU estimation)
        >>> backend = CudaQBackend(simulate=False, estimate_qpu_time=False)
    """

    def __init__(
        self,
        target: str = "qpp-cpu",
        simulate: bool = True,
        estimate_qpu_time: bool = True,
        backend_name: str = "FakeMarrakesh",
        shots: int = 1000,
        optimization_level: int = 3,
    ):
        self._target = target
        self._simulate = simulate
        self._estimate_qpu_time = estimate_qpu_time
        self._backend_name = backend_name
        self._shots = shots
        self._optimization_level = optimization_level
        self._target_set = False

        # Warmup only if we're actually simulating
        self._warmup = simulate

        # Lazy-load fake backend (only when needed for timing)
        self._fake_backend = None
        self._dt = None

        # Per-qtensor cache: qtensor_id -> CompiledQuantumTensor
        self._compiled_cache: dict[int, "CompiledQuantumTensor"] = {}
        # Track compilation times
        self._compilation_times: dict[int, float] = {}
        # Cache for estimated QPU times per qtensor
        self._qpu_time_cache: dict[int, float] = {}

    def _get_fake_backend(self):
        """Lazily initialize the fake backend for timing estimation."""
        if self._fake_backend is None:
            from qiskit_ibm_runtime import fake_provider

            backend_class = getattr(fake_provider, self._backend_name, None)
            if backend_class is None:
                raise ValueError(
                    f"Unknown fake backend: {self._backend_name}. "
                    f"Available: FakeMarrakesh, FakeTorino, FakeBrisbane, etc."
                )
            self._fake_backend = backend_class()
            self._dt = self._fake_backend.configuration().dt
        return self._fake_backend

    @property
    def name(self) -> str:
        mode = "sim" if self._simulate else "nosim"
        return f"cudaq-{self._target}-{mode}"

    @property
    def target(self) -> str:
        """The CUDA-Q target backend."""
        return self._target

    @property
    def simulate(self) -> bool:
        """Whether actual simulation is enabled."""
        return self._simulate

    @property
    def total_compilation_time(self) -> float:
        """Total time spent compiling quantum tensors."""
        return sum(self._compilation_times.values())

    @property
    def total_code_lines(self) -> int:
        """Total number of code lines in all compiled quantum tensors."""
        return sum(compiled.num_code_lines for compiled in self._compiled_cache.values())

    def _ensure_target(self):
        """Set the CUDA-Q target if not already set."""
        if not self._target_set:
            current_target = cudaq.get_target().name
            if current_target != self._target:
                try:
                    cudaq.set_target(self._target)
                except Exception:
                    pass  # Target may not be available
            self._target_set = True

    def prepare(self, qtensors: list["QuantumTensor"]) -> float:
        """Prepare the backend by compiling all quantum tensors.
        
        Args:
            qtensors: List of quantum tensors to prepare.
            
        Returns:
            Total compilation time in seconds.
        """
        start = perf_counter()
        
        # Ensure CudaQ target is set
        self._ensure_target()
        
        for qtensor in qtensors:
            qtensor_id = id(qtensor)
            
            # Compile to CudaQ if not already cached
            if qtensor_id not in self._compiled_cache:
                compile_start = perf_counter()
                self._compiled_cache[qtensor_id] = qtensor.compile(warmup=self._warmup)
                self._compilation_times[qtensor_id] = perf_counter() - compile_start
        
        return perf_counter() - start

    def _estimate_qpu_time_for_qtensor(self, qtensor: "QuantumTensor", params: dict[str, float]) -> float:
        """Estimate QPU time for a quantum tensor using Qiskit scheduling."""
        if not self._estimate_qpu_time:
            return 0.0

        qtensor_id = id(qtensor)

        if qtensor_id in self._qpu_time_cache:
            return self._qpu_time_cache[qtensor_id]

        from qiskit.compiler import transpile
        from qtpu.transforms import remove_operations_by_name

        # Get circuits
        circuits = qtensor.flat()
        num_circuits = len(circuits)

        if num_circuits == 0:
            self._qpu_time_cache[qtensor_id] = 0.0
            return 0.0

        # Take first circuit as representative
        circuit = circuits[0].decompose()

        # Bind parameters
        if circuit.parameters and params:
            circuit_param_names = {p.name for p in circuit.parameters}
            params_to_bind = {
                k: v for k, v in params.items() if k in circuit_param_names
            }
            if params_to_bind:
                circuit = circuit.assign_parameters(params_to_bind)

        # Remove custom operations for transpilation
        circuit = remove_operations_by_name(
            circuit, {"qpd_measure", "iswitch"}, inplace=False
        )

        # Get fake backend and transpile
        fake_backend = self._get_fake_backend()
        scheduled = transpile(
            circuits=circuit,
            backend=fake_backend,
            optimization_level=self._optimization_level,
            scheduling_method="asap",
        )

        # Estimate QPU time
        single_circuit_time = self._estimate_single_circuit_runtime(scheduled)
        estimated_qpu_time = single_circuit_time * num_circuits

        self._qpu_time_cache[qtensor_id] = estimated_qpu_time
        return estimated_qpu_time

    def _estimate_single_circuit_runtime(self, scheduled_circuit) -> float:
        """Estimate QPU runtime for a single scheduled circuit."""
        # Typical values for superconducting qubits
        t_init = 100e-6  # Reset/initialization time
        t_latency = 20e-6  # Classical control latency

        if scheduled_circuit.duration is None:
            # Fallback for circuits without duration info
            t_sched = 1e-6 * scheduled_circuit.depth()
        else:
            t_sched = scheduled_circuit.duration * self._dt

        t_per_shot = t_init + t_sched + t_latency
        return t_per_shot * self._shots

    def evaluate(
        self,
        qtensor: "QuantumTensor",
        params: dict[str, float],
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, float, float]:
        """Evaluate a quantum tensor.

        Returns:
            Tuple of (result, eval_time, estimated_qpu_time).
        """
        qtensor_id = id(qtensor)
        
        # Ensure target is set
        self._ensure_target()

        # Get compiled tensor (compile if needed)
        if qtensor_id not in self._compiled_cache:
            compile_start = perf_counter()
            self._compiled_cache[qtensor_id] = qtensor.compile(warmup=self._warmup)
            self._compilation_times[qtensor_id] = perf_counter() - compile_start
        
        compiled = self._compiled_cache[qtensor_id]

        # Get QPU time estimate if enabled
        estimated_qpu_time = self._estimate_qpu_time_for_qtensor(qtensor, params)

        # Skip actual execution if not simulating
        if not self._simulate:
            result = torch.randn(qtensor.shape, dtype=dtype, device=device)
            return result, 0.0, estimated_qpu_time

        # Execute the compiled tensor with CudaQ
        exec_start = perf_counter()
        result_np = compiled(**params)
        exec_time = perf_counter() - exec_start

        # Convert to torch tensor
        result = torch.tensor(result_np, dtype=dtype, device=device)

        return result, exec_time, estimated_qpu_time

    def sample(
        self,
        qtensor: "QuantumTensor",
        indices: list[tuple[int, ...]],
        params: dict[str, float],
        dtype: torch.dtype = torch.float64,
        device: torch.device | None = None,
    ) -> tuple[list[tuple[tuple[int, ...], float]], float, float]:
        """Sample specific indices from a quantum tensor.
        
        Args:
            qtensor: The quantum tensor to sample from.
            indices: List of index tuples to sample.
            params: Parameter values for the circuit.
            dtype: Data type for results (used if not simulating).
            device: Device for results (used if not simulating).
            
        Returns:
            Tuple of (samples, eval_time, estimated_qpu_time) where:
            - samples: List of (index, value) tuples
            - eval_time: Wall-clock execution time
            - estimated_qpu_time: Estimated QPU time for sampling these indices
        """
        qtensor_id = id(qtensor)
        
        self._ensure_target()

        if qtensor_id not in self._compiled_cache:
            compile_start = perf_counter()
            self._compiled_cache[qtensor_id] = qtensor.compile(warmup=self._warmup)
            self._compilation_times[qtensor_id] = perf_counter() - compile_start
        
        compiled = self._compiled_cache[qtensor_id]
        
        # Estimate QPU time for sampling (proportional to number of indices)
        estimated_qpu_time = self._estimate_qpu_time_for_sampling(qtensor, params, len(indices))
        
        # Skip actual execution if not simulating
        if not self._simulate:
            # Return random values for each index
            samples = [(idx, np.random.randn()) for idx in indices]
            return samples, 0.0, estimated_qpu_time
        
        # Execute sampling
        exec_start = perf_counter()
        samples = compiled.sample(indices=indices, **params)
        exec_time = perf_counter() - exec_start
        
        return samples, exec_time, estimated_qpu_time

    def _estimate_qpu_time_for_sampling(
        self, 
        qtensor: "QuantumTensor", 
        params: dict[str, float],
        num_samples: int
    ) -> float:
        """Estimate QPU time for sampling a specific number of indices.
        
        Unlike full evaluation which runs all indices, sampling only runs
        the specified number of circuits.
        """
        if not self._estimate_qpu_time:
            return 0.0
        
        qtensor_id = id(qtensor)
        
        # Get single circuit time (cached per qtensor)
        if qtensor_id not in self._qpu_time_cache:
            # Force computation of the cache entry via full estimation
            _ = self._estimate_qpu_time_for_qtensor(qtensor, params)
        
        # Get total time for full tensor and compute per-circuit time
        full_num_circuits = int(np.prod(qtensor.shape)) if qtensor.shape else 1
        if qtensor_id in self._qpu_time_cache and full_num_circuits > 0:
            full_time = self._qpu_time_cache[qtensor_id]
            per_circuit_time = full_time / full_num_circuits
            return per_circuit_time * num_samples
        
        return 0.0

    def clear_cache(self):
        """Clear all caches."""
        self._compiled_cache.clear()
        self._compilation_times.clear()
        self._qpu_time_cache.clear()
