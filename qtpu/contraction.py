from itertools import chain
from typing import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy.typing import NDArray
import cotengra as ctg
from qiskit.primitives import Estimator
from qiskit.circuit import QuantumCircuit

from qtpu.tensor import HybridTensorNetwork, QuantumTensor, ClassicalTensor
from qtpu.evaluate import evaluate_estimator


def contract(
    hybrid_tn: HybridTensorNetwork,
    eval_fn: Callable[[list[QuantumCircuit]], list] | None = None,
):
    if eval_fn is None:
        eval_fn = evaluate_estimator(Estimator())

    eq, operators = evaluate(hybrid_tn, eval_fn)
    return ctg.einsum(eq, *operators).item()


def evaluate(
    hybrid_tn: HybridTensorNetwork,
    eval_fn: Callable[[list[QuantumCircuit]], list],
) -> tuple[str, list[NDArray]]:
    quantum_tensors = hybrid_tn.quantum_tensors
    classical_tensors = hybrid_tn.classical_tensors

    serialized_circuits = list(
        chain.from_iterable(
            [circ for circ, _ in qt.instances()] for qt in quantum_tensors
        )
    )

    results = eval_fn(serialized_circuits)

    eval_tensors = []
    for qt in quantum_tensors:
        num_results = np.prod(qt.shape)
        eval_tensors.append(
            ClassicalTensor(
                np.array(results[:num_results], dtype=np.float32).reshape(qt.shape),
                qt.inds,
            )
        )
        results = results[num_results:]

    for ct in eval_tensors:
        print(ct.data)

    return hybrid_tn.equation(), [
        tens.data for tens in eval_tensors + classical_tensors
    ]
