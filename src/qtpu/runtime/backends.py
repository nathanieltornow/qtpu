"""Quantum tensor evaluation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit
    from qiskit.primitives import BaseEstimatorV2
    from qtpu.tensor import QuantumTensor


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
                params_to_bind = {k: v for k, v in params.items() if k in circuit_param_names}
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
        
        # Get circuits
        circuits = qtensor.flat()
        
        # Bind parameters and clean up
        bound_circuits = []
        for circuit in circuits:
            circuit = circuit.decompose()
            if circuit.parameters and params:
                circuit_param_names = {p.name for p in circuit.parameters}
                params_to_bind = {k: v for k, v in params.items() if k in circuit_param_names}
                if params_to_bind:
                    circuit = circuit.assign_parameters(params_to_bind)
            
            # Remove custom operations for transpilation
            circuit = remove_operations_by_name(
                circuit, {"qpd_measure", "iswitch"}, inplace=False
            )
            bound_circuits.append(circuit)
        
        # Transpile and schedule
        scheduled = transpile(
            circuits=bound_circuits,
            backend=self._backend,
            optimization_level=self._optimization_level,
            scheduling_method="asap",
        )
        
        # Estimate QPU time
        estimated_qpu_time = self._estimate_runtime(scheduled)
        
        # Return random results (this is fake!)
        result = torch.randn(qtensor.shape, dtype=dtype, device=device)
        
        eval_time = perf_counter() - start
        return result, eval_time, estimated_qpu_time
    
    def _estimate_runtime(self, scheduled_circuits: list) -> float:
        """Estimate total QPU runtime from scheduled circuits."""
        total_time = 0.0
        
        # Typical values for superconducting qubits
        t_init = 100e-6  # Reset/initialization time
        t_latency = 20e-6  # Classical control latency
        
        for circuit in scheduled_circuits:
            if circuit.duration is None:
                # Fallback for circuits without duration info
                t_sched = 1e-6 * circuit.depth()
            else:
                t_sched = circuit.duration * self._dt
            
            t_per_shot = t_init + t_sched + t_latency
            total_time += t_per_shot * self._shots
        
        return total_time
