from typing import Callable
from itertools import chain

import numpy as np
import quimb.tensor as qtn
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import Estimator

from qtpu.tensor import HybridTensorNetwork, QuantumTensor
from qtpu.evaluate import evaluate_estimator, evaluate_quimb


def contract(
    hybrid_tn: HybridTensorNetwork,
    eval_fn: Callable[[list[QuantumCircuit]], list] | None = None,
) -> qtn.TensorNetwork:

    eval_tn = evaluate_hybrid_tn(hybrid_tn, eval_fn)
    return eval_tn.contract(all, optimize="auto-hq", output_inds=[])


def evaluate_hybrid_tn(
    hybrid_tn: HybridTensorNetwork,
    eval_fn: Callable[[list[QuantumCircuit]], list] | None = None,
) -> qtn.TensorNetwork:

    # if eval_fn is None:
        # eval_fn = evaluate_quimb
    if eval_fn is None:
        eval_fn = evaluate_estimator(Estimator())

    quantum_tensors = hybrid_tn.quantum_tensors
    eval_tensors = evaluate_quantum_tensors(quantum_tensors, eval_fn)
    return qtn.TensorNetwork(eval_tensors + hybrid_tn.qpd_tensors)


def evaluate_quantum_tensors(
    quantum_tensors: list[QuantumTensor],
    eval_fn: Callable[[list[QuantumCircuit]], list],
) -> list[qtn.Tensor]:
    serialized_circuits = list(
        chain.from_iterable([circ for circ in qt.instances()] for qt in quantum_tensors)
    )
    results = eval_fn(serialized_circuits)
    eval_tensors = []
    for qt in quantum_tensors:
        num_results = np.prod(qt.ind_tensor.shape)
        print(num_results)
        eval_tensors.append(
            qtn.Tensor(
                np.array(results[:num_results]).reshape(qt.ind_tensor.shape),
                qt.ind_tensor.inds,
            )
        )
        results = results[num_results:]

    return eval_tensors
