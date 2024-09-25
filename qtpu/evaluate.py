from typing import Callable

import numpy as np
import quimb as qu
import quimb.tensor as qtn
from qiskit.primitives import Estimator, Sampler
from qiskit.providers import Backend
from qiskit.compiler import transpile
from qiskit.circuit import QuantumCircuit, ClassicalRegister

from qtpu.helpers import defer_mid_measurements, qiskit_to_quimb
from qtpu.quasi_distr import QuasiDistr
from qiskit_aer import AerSimulator


def evaluate_estimator(
    estimator: Estimator,
) -> Callable[[list[QuantumCircuit]], list[float]]:
    def _eval(circuits: list[QuantumCircuit]) -> list[float]:
        print(f"Running {len(circuits)} circuits...")
        circuits = [defer_mid_measurements(circ) for circ in circuits]
        observables = [_get_Z_observable(circ) for circ in circuits]
        circuits = [
            circuit.remove_final_measurements(inplace=False) for circuit in circuits
        ]
        results = estimator.run(circuits, observables).result().values
        return list(results)

    return _eval


def evaluate_quimb(circuits: list[QuantumCircuit]):
    circuits = [defer_mid_measurements(circ) for circ in circuits]
    meas_qubits = [_get_meas_qubits(circ) for circ in circuits]
    circuits = [qiskit_to_quimb(circ) for circ in circuits]
    print(circuits)

    results = [
        circuit.local_expectation(_quimb_Z_obs(len(mq)), mq)
        for circuit, mq in zip(circuits, meas_qubits)
    ]
    return results


def _quimb_Z_obs(num_qubits: int):
    Z = qu.pauli("Z")
    for i in range(num_qubits - 1):
        Z = Z & qu.pauli("Z")
    return Z


def evaluate_sampler(
    sampler: Sampler,
    shots: int = 20000,
    return_quasi_distr: bool = False,
):
    def _eval(circuits: list[QuantumCircuit]) -> list[float]:

        cid_withour_meas = [
            i
            for i, circ in enumerate(circuits)
            if circ.count_ops().get("measure", 0) == 0
        ]

        for i, _ in reversed(cid_withour_meas):
            circuits.pop(i)

        # circuits = [
        #     transpile(circ, backend=backend, optimization_level=optimization_level)
        #     for circ in circuits
        # ]

        dists = sampler.run(circuits, shots=shots).result().quasi_dists
        dists = [dists] if isinstance(dists, dict) else dists

        for i in cid_withour_meas:
            dists.insert(i, {0: 1.0})

        if return_quasi_distr:
            size = circuits[0].cregs[0].size
            res = [QuasiDistr(d).prepare(size) for d in dists]

        else:
            res = [d.expval() for d in [QuasiDistr(d) for d in dists]]

        return res

    return _eval


def _evaluate_dummy(circuits: list[QuantumCircuit]) -> list[float]:
    return [np.random.rand() for _ in circuits]


def _get_meas_qubits(circuit: QuantumCircuit) -> list[int]:
    measured_qubits = sorted(
        set(
            circuit.qubits.index(instr.qubits[0])
            for instr in circuit
            if instr.operation.name == "measure"
        ),
    )
    return measured_qubits


def _get_Z_observable(circuit: QuantumCircuit) -> str:
    measured_qubits = _get_meas_qubits(circuit)
    obs = ["I"] * circuit.num_qubits
    for qubit in measured_qubits:
        obs[qubit] = "Z"
    return "".join(reversed(obs))


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
