from typing import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy.typing import NDArray
import cotengra as ctg
from qiskit.primitives import Estimator

from qtpu.tensor import HybridTensorNetwork, QuantumTensor, ClassicalTensor
from qtpu.evaluate import evaluate_estimator


def contract(
    hybrid_tn: HybridTensorNetwork,
    eval_fn: Callable[[QuantumTensor], ClassicalTensor] | None = None,
):
    if eval_fn is None:
        eval_fn = evaluate_estimator(Estimator())

    eq, operators = evaluate(hybrid_tn, eval_fn)
    return ctg.einsum(eq, *operators).item()


def evaluate(
    hybrid_tn: HybridTensorNetwork,
    eval_fn: Callable[[QuantumTensor], ClassicalTensor],
) -> tuple[str, list[NDArray]]:
    quantum_tensors = hybrid_tn.quantum_tensors
    classical_tensors = hybrid_tn.classical_tensors

    with ThreadPoolExecutor() as executor:
        futs = [executor.submit(eval_fn, qt) for qt in quantum_tensors]
        eval_tensors = [fut.result() for fut in futs]

    return hybrid_tn.equation(), [
        tens.data for tens in eval_tensors + classical_tensors
    ]
