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

from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister

from qtpu.runtime.backends import QuantumBackend
from qtpu.transforms import remove_operations_by_name

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit
    from qiskit_ibm_runtime import IBMBackend as IBMBackendType
    from qtpu.core import QuantumTensor


def _defer_qpd_measures(circ):
    """Return a new circuit with each `qpd_measure` op replaced by a
    CNOT(target, fresh_ancilla). Mirrors CudaQ codegen's principle-of-deferred-
    measurement: measuring a qubit in the Z basis mid-circuit is equivalent to
    CNOT-ing it onto a fresh |0⟩ ancilla and measuring the ancilla at the end.
    This preserves the classical outcome of each QPD measurement without
    performing a physical mid-circuit measurement. Non-qpd ops are copied
    verbatim; `iswitch` ops are dropped (they should already be decomposed but
    we strip defensively)."""
    n_qpd = sum(1 for instr in circ.data if instr.operation.name == "qpd_measure")
    if n_qpd == 0:
        # No QPD measurements → just copy the circuit (strip residual iswitches).
        new = QuantumCircuit(*circ.qregs, *circ.cregs, name=circ.name)
        for instr in circ.data:
            if instr.operation.name == "iswitch":
                continue
            new.append(instr.operation, instr.qubits, instr.clbits)
        return new

    anc = QuantumRegister(n_qpd, "qpd_anc")
    new = QuantumCircuit(*circ.qregs, anc, *circ.cregs, name=circ.name)
    anc_idx = 0
    for instr in circ.data:
        if instr.operation.name == "iswitch":
            continue
        if instr.operation.name == "qpd_measure":
            new.cx(instr.qubits[0], anc[anc_idx])
            anc_idx += 1
            continue
        new.append(instr.operation, instr.qubits, instr.clbits)
    return new


def _strip_resets_and_measure(circ) -> None:
    """Mirror the CudaQ codegen trace-out: remove terminal resets entirely and
    measure only non-traced-out qubits. Matches codegen's policy of using the
    I-observable (never emitting the physical reset) for qubits the wire cut
    or caller marked as traced-out.

    A reset is terminal if no further non-barrier op touches the same qubit.
    Terminal resets are deleted in-place and their qubits are omitted from
    the measurement register (I in the Z^n product). Mid-circuit resets —
    resets followed by more ops on the same qubit — would represent a
    deferred mid-circuit measurement and need ancilla expansion to be HW-safe;
    we raise rather than submit a noisy mid-circuit measurement."""
    last_op_idx: dict[int, int] = {}
    last_op_name: dict[int, str] = {}
    for idx, instr in enumerate(circ.data):
        if instr.operation.name == "barrier":
            continue
        for q in instr.qubits:
            qi = circ.qubits.index(q)
            last_op_idx[qi] = idx
            last_op_name[qi] = instr.operation.name

    # Detect mid-circuit resets: any reset whose qubit has a later non-barrier op.
    for idx, instr in enumerate(circ.data):
        if instr.operation.name != "reset":
            continue
        for q in instr.qubits:
            qi = circ.qubits.index(q)
            if last_op_idx[qi] > idx:
                raise NotImplementedError(
                    f"Mid-circuit reset on qubit {qi} at position {idx} "
                    "(followed by more ops on the same qubit). Not supported — "
                    "requires deferred-measurement ancilla expansion to be HW-safe."
                )

    # Delete terminal resets in reverse index order to keep indices valid.
    terminal_reset_indices = sorted(
        {last_op_idx[qi] for qi, nm in last_op_name.items() if nm == "reset"},
        reverse=True,
    )
    for idx in terminal_reset_indices:
        del circ.data[idx]

    # Measure everything that wasn't terminal-reset (= I-observable).
    to_measure = [qi for qi in range(circ.num_qubits) if last_op_name.get(qi) != "reset"]
    if not to_measure:
        return
    creg = ClassicalRegister(len(to_measure), "meas")
    circ.add_register(creg)
    for j, qi in enumerate(to_measure):
        circ.measure(circ.qubits[qi], creg[j])


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

        # Clean custom ops, defer QPD measurements to ancillas, strip terminal
        # resets, and measure non-traced-out qubits + ancillas. Mirrors the
        # CudaQ codegen policy: (a) no physical reset (I-observable for traced-
        # out qubits), (b) each qpd_measure becomes CNOT(target, fresh ancilla)
        # and the ancilla is measured at the end to preserve the QPD outcome.
        # Split into "needs submission" (has at least one measurement after
        # cleaning) and "trivial" (identity observable — empty measurement
        # register). Trivial variants arise when the fragment contains no
        # observable-support qubits AND no QPD ancillas (prep-side fragments
        # of cuts that don't include the observed qubit). Their expval is
        # tr[ρ_k] = 1 for every prep variant k, so we skip the hardware submit.
        clean = []
        trivial_flags = []
        for circ in bound:
            c = _defer_qpd_measures(circ)
            if not any(i.operation.name == "measure" for i in c.data):
                _strip_resets_and_measure(c)
            has_meas = any(i.operation.name == "measure" for i in c.data)
            trivial_flags.append(not has_meas)
            if has_meas:
                clean.append(c)

        # Transpile for hardware
        if clean:
            transpiled = transpile(
                clean,
                backend=self._backend,
                optimization_level=self._optimization_level,
            )
            # Submit to IBM
            sampler = SamplerV2(mode=self._backend)
            job = sampler.run(transpiled, shots=self._shots)
            result = job.result()
            metrics = job.metrics()
            actual_qpu_time = metrics.get("usage", {}).get("quantum_seconds", 0.0)
            self._total_actual_qpu_time += actual_qpu_time
            self._total_jobs += 1
        else:
            result = []
            actual_qpu_time = 0.0

        # Convert measurement results to tensor values.
        # For each circuit, compute the expval of the Z⊗...⊗Z parity over the
        # measured qubits. Trivial (no-measurement) variants contribute 1.0.
        values = []
        result_iter = iter(result)
        for is_trivial in trivial_flags:
            if is_trivial:
                values.append(1.0)
                continue
            pub_result = next(result_iter)
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
