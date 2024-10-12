from typing import Callable
from itertools import chain

import numpy as np
import quimb.tensor as qtn
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import BaseEstimatorV2
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2

from qtpu.tensor import HybridTensorNetwork, QuantumTensor
from qtpu.helpers import defer_mid_measurements
from qtpu.quasi_distr import QuasiDistr


def contract(
    hybrid_tn: HybridTensorNetwork,
    evaluator: BaseEstimatorV2 | AerSimulator | None = None,
) -> qtn.TensorNetwork:

    eval_tn = evaluate_hybrid_tn(hybrid_tn, evaluator)
    return eval_tn.contract(all, optimize="auto-hq", output_inds=[])


def evaluate_hybrid_tn(
    hybrid_tn: HybridTensorNetwork,
    evaluator: BaseEstimatorV2 | AerSimulator | None = None,
) -> qtn.TensorNetwork:

    if evaluator is None:
        evaluator = EstimatorV2()

    quantum_tensors = hybrid_tn.quantum_tensors
    eval_tensors = _evaluate_quantum_tensors(quantum_tensors, evaluator)
    return qtn.TensorNetwork(eval_tensors + hybrid_tn.qpd_tensors)


def _evaluate_quantum_tensors(
    quantum_tensors: list[QuantumTensor], evaluator: BaseEstimatorV2
) -> list[qtn.Tensor]:

    print(f"Evaluating {sum(qt.ind_tensor.size for qt in quantum_tensors)} circuits")

    serialized_circuits = list(
        chain.from_iterable([circ for circ in qt.instances()] for qt in quantum_tensors)
    )

    if isinstance(evaluator, BaseEstimatorV2):
        results = _evaluate_estimator(evaluator, serialized_circuits)
    elif isinstance(evaluator, AerSimulator):
        results = _evaluate_simulator(evaluator, serialized_circuits)
    else:
        raise ValueError("Invalid evaluator")

    eval_tensors = []
    for qt in quantum_tensors:
        num_results = np.prod(qt.ind_tensor.shape)
        eval_tensors.append(
            qtn.Tensor(
                np.array(results[:num_results]).reshape(qt.ind_tensor.shape),
                qt.ind_tensor.inds,
            )
        )
        results = results[num_results:]

    return eval_tensors


def _evaluate_estimator(
    estimator: BaseEstimatorV2, circuits: list[QuantumCircuit]
) -> Callable[[list[QuantumCircuit]], list[float]]:
    circuits = [defer_mid_measurements(circ) for circ in circuits]
    observables = [_get_Z_observable(circ) for circ in circuits]
    circuits = [
        circuit.remove_final_measurements(inplace=False) for circuit in circuits
    ]
    results = estimator.run(list(zip(circuits, observables))).result()
    expvals = [res.data.evs for res in results]
    return expvals


def _evaluate_simulator(
    sim: AerSimulator,
    circuits: list[QuantumCircuit],
):
    cid_withour_meas = [
        i for i, circ in enumerate(circuits) if circ.count_ops().get("measure", 0) == 0
    ]

    for i in reversed(cid_withour_meas):
        circuits.pop(i)

    counts = (
        sim.run([defer_mid_measurements(circ) for circ in circuits])
        .result()
        .get_counts()
    )
    counts = [counts] if isinstance(counts, dict) else counts

    for i in cid_withour_meas:
        counts.insert(i, {"0": 20000})

    # we assume that the first register is the original register of the original circuit
    # size = circuits[0].cregs[0].size
    # res = [QuasiDistr(d).expval(size) for d in dists]

    res = [QuasiDistr.from_counts(d).expval() for d in counts]
    return res


def _get_Z_observable(circuit: QuantumCircuit) -> str:
    measured_qubits = _get_meas_qubits(circuit)
    obs = ["I"] * circuit.num_qubits
    for qubit in measured_qubits:
        obs[qubit] = "Z"
    return "".join(reversed(obs))


def _get_meas_qubits(circuit: QuantumCircuit) -> list[int]:
    measured_qubits = sorted(
        set(
            circuit.qubits.index(instr.qubits[0])
            for instr in circuit
            if instr.operation.name == "measure"
        ),
    )
    return measured_qubits
