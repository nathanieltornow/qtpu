"""Quantum tensor evaluation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import torch
import cudaq

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit
    from qiskit.primitives import BaseEstimatorV2
    from qtpu.core import QuantumTensor
    from qtpu.compiler.codegen import CompiledQuantumTensor


class QuantumBackend(ABC):
    """Abstract base class for quantum tensor evaluation backends.

    Subclass this to create custom backends (e.g., for real QPU hardware).

    Example:
        >>> class MyCloudBackend(QuantumBackend):
        ...     @property
        ...     def name(self) -> str:
        ...         return "my_cloud"
        ...
        ...     def evaluate(self, qtensor, params, dtype, device):
        ...         # Submit to cloud, wait for results
        ...         ...
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name for logging."""
        ...

    def prepare(
        self,
        qtensors: list["QuantumTensor"],
    ) -> float:
        """Prepare the backend for evaluating the given quantum tensors.
        
        This is called during HEinsumRuntime.prepare() and allows backends
        to pre-compile circuits, estimate QPU times, etc.
        
        Args:
            qtensors: List of quantum tensors that will be evaluated.
            
        Returns:
            Total preparation/compilation time in seconds.
        """
        # Default implementation: no preparation needed
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

        Args:
            qtensor: The quantum tensor to evaluate.
            params: Circuit parameters (rotation angles, etc.).
            dtype: Output tensor dtype.
            device: Output tensor device.

        Returns:
            Tuple of:
            - result: Evaluated tensor with shape qtensor.shape
            - eval_time: Wall-clock evaluation time in seconds
            - estimated_qpu_time: Estimated real QPU time (0 for pure simulators)
        """
        ...


class SimulatorBackend(QuantumBackend):
    """Statevector/density matrix simulation backend using Qiskit Aer.

    This backend provides exact simulation of quantum circuits, suitable for
    development and small-scale testing.

    Args:
        estimator: Qiskit estimator primitive. If None, uses Aer EstimatorV2.

    Example:
        >>> backend = SimulatorBackend()
        >>> result, eval_time, _ = backend.evaluate(qtensor, {}, torch.float64, torch.device("cpu"))
    """

    def __init__(self, estimator: "BaseEstimatorV2 | None" = None):
        if estimator is None:
            from qiskit_aer.primitives import EstimatorV2

            estimator = EstimatorV2()
        self._estimator = estimator

    @property
    def name(self) -> str:
        return "simulator"

    def evaluate(
        self,
        qtensor: "QuantumTensor",
        params: dict[str, float],
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, float, float]:
        from qtpu.transforms import decompose_qpd_measures, remove_operations_by_name

        start = perf_counter()

        circuits = qtensor.flat()
        bound_circuits = []
        observables = []

        for circuit in circuits:
            circuit = circuit.decompose()

            # Bind parameters
            if circuit.parameters and params:
                circuit_param_names = {p.name for p in circuit.parameters}
                params_to_bind = {
                    k: v for k, v in params.items() if k in circuit_param_names
                }
                if params_to_bind:
                    circuit = circuit.assign_parameters(params_to_bind)

            # Handle QPD measures
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
        results = self._estimator.run(jobs).result()
        expvals = [r.data.evs for r in results]

        # Reshape and convert
        data = np.array(expvals).reshape(qtensor.shape)
        result = torch.tensor(data, dtype=dtype, device=device)

        eval_time = perf_counter() - start
        return result, eval_time, 0.0

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


class FakeQPUBackend(QuantumBackend):
    """Fake QPU backend that estimates timing and returns random results.

    This backend:
    - Transpiles circuits to a real backend topology
    - Schedules circuits to estimate actual QPU execution time
    - Returns random results (for timing evaluation only)

    Useful for scalability studies without running actual quantum hardware.

    Args:
        backend_name: Name of the fake backend (e.g., "FakeMarrakesh", "FakeTorino").
        shots: Number of shots for time estimation.
        optimization_level: Transpilation optimization level (0-3).

    Example:
        >>> backend = FakeQPUBackend("FakeMarrakesh", shots=1000)
        >>> result, eval_time, qpu_time = backend.evaluate(qtensor, {}, torch.float64, device)
        >>> print(f"Estimated QPU time: {qpu_time:.3f}s")
    """

    def __init__(
        self,
        backend_name: str = "FakeMarrakesh",
        shots: int = 1000,
        optimization_level: int = 3,
    ):
        self._shots = shots
        self._optimization_level = optimization_level
        self._backend = self._get_backend(backend_name)
        self._dt = self._backend.configuration().dt
        self._backend_name = backend_name

    def _get_backend(self, name: str):
        """Get fake backend by name."""
        from qiskit_ibm_runtime import fake_provider

        backend_class = getattr(fake_provider, name, None)
        if backend_class is None:
            raise ValueError(
                f"Unknown fake backend: {name}. "
                f"Available: FakeMarrakesh, FakeTorino, FakeBrisbane, etc."
            )
        return backend_class()

    @property
    def name(self) -> str:
        return "fake_qpu"

    @property
    def backend_name(self) -> str:
        """Name of the underlying fake backend."""
        return self._backend_name

    @property
    def shots(self) -> int:
        """Number of shots used for timing estimation."""
        return self._shots

    def evaluate(
        self,
        qtensor: "QuantumTensor",
        params: dict[str, float],
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, float, float]:
        from qiskit.compiler import transpile
        from qtpu.transforms import remove_operations_by_name

        start = perf_counter()

        # Get circuits - we only need the first one since all have same structure
        circuits = qtensor.flat()
        num_circuits = len(circuits)

        if num_circuits == 0:
            return torch.empty(qtensor.shape, dtype=dtype, device=device), 0.0, 0.0

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

        # Transpile and schedule just the representative circuit
        scheduled = transpile(
            circuits=circuit,
            backend=self._backend,
            optimization_level=self._optimization_level,
            scheduling_method="asap",
        )

        # Estimate QPU time for one circuit, then multiply by count
        single_circuit_time = self._estimate_single_circuit_runtime(scheduled)
        estimated_qpu_time = single_circuit_time * num_circuits

        # Return random results (this is fake!)
        result = torch.randn(qtensor.shape, dtype=dtype, device=device)

        eval_time = perf_counter() - start
        return result, eval_time, estimated_qpu_time

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


class FakeQPUCudaQBackend(QuantumBackend):
    """Hybrid backend: CudaQ JIT compilation + Fake QPU timing estimation.

    This backend combines the best of both worlds:
    - Uses CudaQ JIT compilation for fast, accurate circuit simulation
    - Estimates realistic QPU execution time using Qiskit scheduling

    This is useful for benchmarking where you want:
    - Fast execution (via CudaQ JIT)
    - Realistic QPU time estimates (via Qiskit scheduling on fake backends)

    The recommended workflow is:
    1. Create the backend
    2. Call prepare() with quantum tensors (compiles circuits, estimates QPU times)
    3. Call evaluate() for fast execution

    Args:
        backend_name: Name of the fake backend for timing (e.g., "FakeMarrakesh").
        target: CUDA-Q target for simulation.
        shots: Number of shots for time estimation.
        optimization_level: Transpilation optimization level (0-3).
        warmup: Whether to warmup CudaQ JIT on first compile.

    Example:
        >>> backend = FakeQPUCudaQBackend("FakeMarrakesh", target="nvidia")
        >>> backend.prepare(qtensors)  # Compile all circuits upfront
        >>> result, eval_time, qpu_time = backend.evaluate(qtensor, {}, ...)
    """

    def __init__(
        self,
        backend_name: str = "FakeMarrakesh",
        target: str = "qpp-cpu",
        shots: int = 1000,
        optimization_level: int = 3,
        warmup: bool = True,
    ):
        self._backend_name = backend_name
        self._target = target
        self._shots = shots
        self._optimization_level = optimization_level
        self._warmup = warmup
        self._target_set = False

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
        return f"fake_qpu_cudaq-{self._target}"

    @property
    def target(self) -> str:
        """The CUDA-Q target backend."""
        return self._target

    @property
    def backend_name(self) -> str:
        """Name of the underlying fake backend for timing."""
        return self._backend_name

    @property
    def shots(self) -> int:
        """Number of shots used for timing estimation."""
        return self._shots

    @property
    def total_compilation_time(self) -> float:
        """Total time spent compiling quantum tensors."""
        return sum(self._compilation_times.values())

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

    def prepare(
        self,
        qtensors: list["QuantumTensor"],
    ) -> float:
        """Prepare the backend by compiling all quantum tensors and estimating QPU times.
        
        This does all the heavy lifting upfront:
        1. Sets the CudaQ target
        2. Compiles each quantum tensor to CudaQ kernels
        3. Estimates QPU execution time for each tensor
        
        Args:
            qtensors: List of quantum tensors to prepare.
            
        Returns:
            Total preparation time in seconds.
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
            
            # Estimate QPU time if not already cached
            if qtensor_id not in self._qpu_time_cache:
                self._estimate_qpu_time(qtensor, {})
        
        return perf_counter() - start

    def _estimate_qpu_time(self, qtensor: "QuantumTensor", params: dict[str, float]) -> float:
        """Estimate QPU time for a quantum tensor using Qiskit scheduling.

        This is cached per-qtensor since the circuit structure doesn't change.
        """
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

        If prepare() was called, this is fast (uses cached compiled tensor).
        If not, compilation happens on first call (slower).

        Returns:
            Tuple of (result, eval_time, estimated_qpu_time).
        """
        qtensor_id = id(qtensor)
        
        # Ensure target is set
        self._ensure_target()

        # Get compiled tensor (compile if needed - but should be cached from prepare())
        if qtensor_id not in self._compiled_cache:
            compile_start = perf_counter()
            self._compiled_cache[qtensor_id] = qtensor.compile(warmup=self._warmup)
            self._compilation_times[qtensor_id] = perf_counter() - compile_start
        
        compiled = self._compiled_cache[qtensor_id]

        # Get QPU time estimate (should be cached from prepare())
        estimated_qpu_time = self._estimate_qpu_time(qtensor, params)

        # Execute the compiled tensor with CudaQ
        exec_start = perf_counter()
        result_np = compiled(**params)
        exec_time = perf_counter() - exec_start

        # Convert to torch tensor
        result = torch.tensor(result_np, dtype=dtype, device=device)

        return result, exec_time, estimated_qpu_time

    def clear_cache(self):
        """Clear all caches (compiled tensors and QPU time estimates)."""
        self._compiled_cache.clear()
        self._compilation_times.clear()
        self._qpu_time_cache.clear()


class CudaQBackend(QuantumBackend):
    """CUDA-Q backend for fast quantum tensor evaluation.

    Uses NVIDIA CUDA-Q for JIT-compiled quantum circuit simulation.
    Caches compiled tensors per-qtensor for fast repeated evaluation.

    Requires cuda-quantum to be installed:
        pip install cuda-quantum-cu12  # For CUDA 12

    Args:
        target: CUDA-Q target backend. Options include:
            - "qpp-cpu" (default): CPU simulation with OpenMP
            - "nvidia": GPU simulation (requires NVIDIA GPU)
            - "nvidia-fp64": GPU with double precision
            - "nvidia-mqpu": Multi-GPU simulation
            - "tensornet": Tensor network simulation
        warmup: Whether to run a warmup execution after compilation.

    Example:
        >>> backend = CudaQBackend(target="nvidia")
        >>> backend.prepare(qtensors)  # Compile all circuits upfront
        >>> result, eval_time, _ = backend.evaluate(qtensor, {}, torch.float64, device)

    Note:
        Call prepare() before evaluate() for best performance.
        If not called, compilation happens on first evaluate() call.
    """

    def __init__(self, target: str = "qpp-cpu", warmup: bool = True):
        self._target = target
        self._target_set = False
        # Per-qtensor cache: qtensor_id -> CompiledQuantumTensor
        self._compiled_cache: dict[int, "CompiledQuantumTensor"] = {}
        # Track compilation times
        self._compilation_times: dict[int, float] = {}
        self._warmup = warmup

    @property
    def name(self) -> str:
        return f"cudaq-{self._target}"

    @property
    def target(self) -> str:
        """The CUDA-Q target backend."""
        return self._target

    @property
    def total_compilation_time(self) -> float:
        """Total time spent compiling quantum tensors."""
        return sum(self._compilation_times.values())

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

    def prepare(
        self,
        qtensors: list["QuantumTensor"],
    ) -> float:
        """Prepare the backend by compiling all quantum tensors.
        
        This compiles each quantum tensor to CudaQ kernels upfront,
        so that evaluate() calls are fast.
        
        Args:
            qtensors: List of quantum tensors to compile.
            
        Returns:
            Total compilation time in seconds.
        """
        start = perf_counter()
        
        # Ensure CudaQ target is set
        self._ensure_target()
        
        for qtensor in qtensors:
            qtensor_id = id(qtensor)
            
            # Compile if not already cached
            if qtensor_id not in self._compiled_cache:
                compile_start = perf_counter()
                self._compiled_cache[qtensor_id] = qtensor.compile(warmup=self._warmup)
                self._compilation_times[qtensor_id] = perf_counter() - compile_start
        
        return perf_counter() - start

    def evaluate(
        self,
        qtensor: "QuantumTensor",
        params: dict[str, float],
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, float, float]:
        """Evaluate a quantum tensor.

        If prepare() was called, this is fast (uses cached compiled tensor).
        If not, compilation happens on first call.

        Returns:
            Tuple of (result, eval_time, qpu_time).
            qpu_time is always 0 for this simulator backend.
        """
        qtensor_id = id(qtensor)
        
        # Ensure target is set
        self._ensure_target()

        # Get compiled tensor (compile if needed - but should be cached from prepare())
        if qtensor_id not in self._compiled_cache:
            compile_start = perf_counter()
            self._compiled_cache[qtensor_id] = qtensor.compile(warmup=self._warmup)
            self._compilation_times[qtensor_id] = perf_counter() - compile_start
        
        compiled = self._compiled_cache[qtensor_id]

        # Execute the compiled tensor
        exec_start = perf_counter()
        result_np = compiled(**params)
        exec_time = perf_counter() - exec_start

        # Convert to torch tensor
        result = torch.tensor(result_np, dtype=dtype, device=device)

        return result, exec_time, 0.0

    def clear_cache(self):
        """Clear the compiled tensor cache."""
        self._compiled_cache.clear()
        self._compilation_times.clear()
