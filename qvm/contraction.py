from concurrent.futures import ThreadPoolExecutor

import cotengra as ctg

from qvm.tensor import HybridTensorNetwork
from qvm.evaluator import QTensorEvaluator


def contract(
    hybrid_tn: HybridTensorNetwork, qtensor_evaluator: QTensorEvaluator, shots: int
) -> HybridTensorNetwork:

    quantum_tensors = hybrid_tn.quantum_tensors
    classical_tensors = hybrid_tn.classical_tensors

    with ThreadPoolExecutor() as executor:
        futs = [
            executor.submit(qtensor_evaluator.evaluate_qtensor, qtens, shots=shots)
            for qtens in quantum_tensors
        ]
        eval_tensors = [fut.result() for fut in futs]

    return ctg.einsum(
        hybrid_tn.equation(), *[tens._data for tens in eval_tensors + classical_tensors]
    )
