from concurrent.futures import ThreadPoolExecutor
from typing import Any

import quimb.tensor as qtn
import cotengra as ctg

from qvm.runtime.qpu_manager import QPUManager
from qvm.tensor import QuantumTensor, HybridTensorNetwork


def contract_hybrid_tn(
    hybrid_tn: HybridTensorNetwork,
    qpu_manager: QPUManager,
    qtens_kwarg_map: list[dict[str, Any]] | None = None,
) -> float | qtn.Tensor:

    quantum_tensors = hybrid_tn.quantum_tensors
    classical_tensors = hybrid_tn.classical_tensors

    if qtens_kwarg_map is None:
        qtens_kwarg_map = [{}] * len(quantum_tensors)

    assert len(quantum_tensors) == len(qtens_kwarg_map)

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                _process_quantum_tensor,
                qpu_manager,
                qtens,
                **qtens_kwargs,
            )
            for qtens, qtens_kwargs in zip(quantum_tensors, qtens_kwarg_map)
        ]

        # while the circuits are running, compute a contraction tree
        contr_future = executor.submit(_get_optimized_contraction_tree, hybrid_tn)

        tensors = [fut.result() for fut in futures]
        contr_tree = contr_future.result()

    tn = qtn.TensorNetwork(tensors + classical_tensors)
    return tn.contract(all, optimize=contr_tree)


def _process_quantum_tensor(
    qpu_manager: QPUManager, quantum_tensor: QuantumTensor, **kwargs
) -> qtn.Tensor:
    return qpu_manager.run_quantum_tensor(quantum_tensor, **kwargs)


def _get_optimized_contraction_tree(
    hybrid_tn: HybridTensorNetwork,
) -> ctg.ContractionTree:
    opt = ctg.HyperOptimizer()
    return opt.search(hybrid_tn.inputs(), hybrid_tn.output(), hybrid_tn.size_dict())
