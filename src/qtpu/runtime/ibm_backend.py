"""IBM Quantum backend for quantum tensor evaluation.

Runs subcircuits on real IBM hardware via qiskit-ibm-runtime SamplerV2,
implementing the same QuantumBackend interface as CudaQBackend.
"""

from __future__ import annotations

import os
from itertools import product
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import torch
from qiskit.compiler import transpile

from qtpu.runtime.backends import QuantumBackend
from qtpu.transforms import remove_operations_by_name

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit
    from qiskit_ibm_runtime import IBMBackend as IBMBackendType
    from qtpu.core import QuantumTensor


class IBMBackend(QuantumBackend):
    """Backend that runs quantum tensors on real IBM hardware.

    Each quantum tensor is expanded into flat circuits (one per index
    combination), transpiled, and submitted to IBM via SamplerV2.
    Results are collected into a tensor matching the quantum tensor shape.

    Args:
        backend: An IBM backend instance from QiskitRuntimeService,
            or a backend name string (requires IBM_TOKEN and IBM_CRN env vars).
        shots: Number of shots per circuit.
        optimization_level: Transpilation optimization level (0-3).

    Example:
        >>> from qiskit_ibm_runtime import QiskitRuntimeService
        >>> service = QiskitRuntimeService(channel="ibm_cloud", ...)
        >>> backend = IBMBackend(service.backend("ibm_marrakesh"))
        >>> runtime = HEinsumRuntime(heinsum, backend=backend)
        >>> result, timing = runtime.execute()
    """

    def __init__(
        self,
        backend: IBMBackendType | str = "ibm_marrakesh",
        shots: int = 1000,
        optimization_level: int = 3,
    ):
        if isinstance(backend, str):
            backend = self._connect(backend)
        self._backend = backend
        self._shots = shots
        self._optimization_level = optimization_level
        self._total_actual_qpu_time = 0.0
        self._total_jobs = 0

    @staticmethod
    def _connect(backend_name: str):
        """Connect to IBM Quantum and get the backend."""
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService(
            channel="ibm_cloud",
            token=os.environ["IBM_TOKEN"],
            instance=os.environ["IBM_CRN"],
        )
        return service.backend(backend_name)

    @property
    def name(self) -> str:
        return f"ibm-{self._backend.name}"

    @property
    def total_actual_qpu_time(self) -> float:
        """Total actual QPU time across all jobs (from IBM metadata)."""
        return self._total_actual_qpu_time

    @property
    def total_jobs(self) -> int:
        """Total number of jobs submitted."""
        return self._total_jobs

    def evaluate(
        self,
        qtensor: QuantumTensor,
        params: dict[str, float],
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, float, float]:
        """Evaluate a quantum tensor on IBM hardware.

        Expands the quantum tensor into flat circuits, transpiles them,
        runs on hardware, and collects measurement results into a tensor.

        Returns:
            Tuple of (result_tensor, eval_time, actual_qpu_time).
        """
        from qiskit_ibm_runtime import SamplerV2

        eval_start = perf_counter()

        # Get all flat circuits
        flat_circuits = qtensor.flat()
        n_circuits = len(flat_circuits)

        if n_circuits == 0:
            result = torch.zeros(qtensor.shape, dtype=dtype, device=device)
            return result, 0.0, 0.0

        # Decompose ISwitches into concrete gates
        decomposed = [c.decompose() for c in flat_circuits]

        # Bind parameters
        bound = []
        for circ in decomposed:
            if circ.parameters and params:
                param_names = {p.name for p in circ.parameters}
                to_bind = {k: v for k, v in params.items() if k in param_names}
                if to_bind:
                    circ = circ.assign_parameters(to_bind)
            bound.append(circ)

        # Clean custom ops and add measurements
        clean = []
        for circ in bound:
            c = remove_operations_by_name(circ, {"qpd_measure", "iswitch"}, inplace=False)
            if not any(i.operation.name == "measure" for i in c.data):
                c.measure_all()
            clean.append(c)

        # Transpile for hardware
        transpiled = transpile(
            clean,
            backend=self._backend,
            optimization_level=self._optimization_level,
        )

        # Submit to IBM
        sampler = SamplerV2(mode=self._backend)
        job = sampler.run(transpiled, shots=self._shots)
        result = job.result()

        # Extract actual QPU time
        metrics = job.metrics()
        actual_qpu_time = metrics.get("usage", {}).get("quantum_seconds", 0.0)
        self._total_actual_qpu_time += actual_qpu_time
        self._total_jobs += 1

        # Convert measurement results to tensor values
        # For each circuit, compute the expectation value of the Z⊗Z⊗...⊗Z observable
        values = []
        for pub_result in result:
            counts = pub_result.data.meas.get_counts()
            total_shots = sum(counts.values())
            expval = 0.0
            for bitstring, count in counts.items():
                # Parity: +1 if even number of 1s, -1 if odd
                parity = (-1) ** bitstring.count("1")
                expval += parity * count / total_shots
            values.append(expval)

        eval_time = perf_counter() - eval_start

        # Reshape into tensor
        result_tensor = torch.tensor(values, dtype=dtype, device=device)
        if qtensor.shape:
            result_tensor = result_tensor.reshape(qtensor.shape)

        return result_tensor, eval_time, actual_qpu_time
