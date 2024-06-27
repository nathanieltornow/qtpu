from typing import Callable

import numpy as np
from qiskit.primitives import Estimator
from qiskit.providers import Backend
from qiskit.compiler import transpile
from qiskit.circuit import QuantumCircuit, ClassicalRegister

from qtpu.tensor import QuantumTensor, ClassicalTensor
from qtpu.quasi_distr import QuasiDistr


def evaluate_estimator(
    estimator: Estimator,
) -> Callable[[QuantumTensor], ClassicalTensor]:
    def _eval(qt: QuantumTensor) -> ClassicalTensor:
        circuits = [circ.copy() for circ, _ in qt.instances()]
        observables = [_get_Z_observable(circ) for circ in circuits]
        if not all(circuit.num_qubits == len(obs) for circuit, obs in zip(circuits, observables)):
            raise ValueError("Circuit and observable qubit count mismatch")
        print(circuits[0])
        results = np.array([
            estimator.run(circ, obs).result().values[0] for circ, obs in zip(circuits, observables)
        ]).reshape(qt.shape)
        # results = estimator.run(circuits, observables).result().values.reshape(qt.shape)
        return ClassicalTensor(results, inds=qt.inds)

    return _eval


def evaluate_backend(
    backend: Backend, shots: int = 10000, optimization_level: int = 0
) -> Callable[[QuantumTensor], ClassicalTensor]:
    def _eval(qt: QuantumTensor) -> ClassicalTensor:
        circuits = [circ for circ, _ in qt.instances()]

        cid_withour_meas = [
            (i, s)
            for i, circ in enumerate(circuits)
            if circ.count_ops().get("measure", 0) == 0
        ]

        for i, _ in reversed(cid_withour_meas):
            circuits.pop(i)

        circuits = [
            transpile(circ, backend=backend, optimization_level=optimization_level)
            for circ in circuits
        ]

        counts = backend.run(circuits, shots=shots).result().get_counts()
        counts = [counts] if isinstance(counts, dict) else counts

        for i, s in cid_withour_meas:
            counts.insert(i, {"0": s})

        quasi_dists = np.array(
            [
                QuasiDistr.from_counts(count).prepare(qt.circuit.num_clbits)
                for count in counts
            ]
        ).reshape(qt.shape)

        return ClassicalTensor(quasi_dists, inds=qt.inds)

    return _eval


def _get_Z_observable(circuit: QuantumCircuit) -> str:
    measured_qubits = sorted(
        set(
            circuit.qubits.index(instr.qubits[0])
            for instr in circuit
            if instr.operation.name == "measure"
        ),
    )
    obs = ["I"] * circuit.num_qubits
    for qubit in measured_qubits:
        obs[qubit] = "Z"
    # circuit.remove_final_measurements()
    return "".join(obs)


def remap_circuit(circuit: QuantumCircuit) -> tuple[QuantumCircuit, list[int]]:
    active_clbit_indices = sorted(
        set(circuit.clbits.index(clbit) for instr in circuit for clbit in instr.clbits),
    )
    mapping = {clbit: i for i, clbit in enumerate(active_clbit_indices)}

    new_creg = ClassicalRegister(len(active_clbit_indices), name="c")
    new_circuit = QuantumCircuit(*circuit.qregs, new_creg)
    for instr in circuit:
        new_circuit.append(
            instr.operation,
            instr.qubits,
            [mapping[circuit.clbits.index(clbit)] for clbit in instr.clbits],
        )

    return new_circuit, active_clbit_indices


def remap_result(bitstr: str, num_clbits: int, active_indices: list[int]) -> str:
    assert len(bitstr) <= num_clbits
    bits = ["0"] * num_clbits
    for i, idx in enumerate(active_indices):
        bits[-idx - 1] = bitstr[i]
    return "".join(bits)
