from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy.typing import NDArray
import cotengra as ctg

from qvm.tensor import HybridTensorNetwork, QuantumTensor, ClassicalTensor

from qvm.qinterface import QuantumInterface, BackendInterface


def contract(
    hybrid_tn: HybridTensorNetwork,
    shots: int,
    qiface: QuantumInterface | None = None,
    result_type: str = "expval",
):
    if qiface is None:
        qiface = BackendInterface()
    eq, operators = evaluate(hybrid_tn, shots, qiface, result_type)
    return ctg.einsum(eq, *operators)


def evaluate(
    hybrid_tn: HybridTensorNetwork,
    shots: int,
    qiface: QuantumInterface,
    result_type: str,
) -> tuple[str, list[NDArray]]:
    quantum_tensors = hybrid_tn.quantum_tensors
    classical_tensors = hybrid_tn.classical_tensors

    with ThreadPoolExecutor() as executor:
        futs = [
            executor.submit(evaluate_quantum_tensor, qtens, qiface, shots, result_type)
            for qtens in quantum_tensors
        ]
        eval_tensors = [fut.result() for fut in futs]

    return hybrid_tn.equation(), [
        tens.data for tens in eval_tensors + classical_tensors
    ]


def evaluate_quantum_tensor(
    qtensor: QuantumTensor,
    qiface: QuantumInterface,
    shots: int,
    result_type: str,
) -> ClassicalTensor:
    num_bits = qtensor.circuit.cregs[0].size

    circuits = [circ for circ, _ in qtensor.instances()]
    shot_per_circ = [int(shots * weight) for _, weight in qtensor.instances()]

    quasi_dists = qiface.run(circuits, shot_per_circ)

    match result_type:
        case "expval":
            arr = np.array([qd.expval() for qd in quasi_dists])
        case "samples":
            arr = np.array([qd.prepare(num_bits) for qd in quasi_dists])

    return ClassicalTensor(arr.reshape(qtensor.shape), qtensor.inds)
