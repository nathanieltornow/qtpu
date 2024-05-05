import quimb.tensor as qtn

from qvm.qpu_manager import QPUManager
from qvm.tensor import QuantumTensor, HybridTensorNetwork


def contract_hybrid_tn(
    hybrid_tn: HybridTensorNetwork, qpu_manager: QPUManager
) -> float:

    classical_tensors = hybrid_tn._classical_tensors.copy()
    quantum_tensors = hybrid_tn._quantum_tensors

    for quantum_tensor in quantum_tensors:
        tensor = qpu_manager.run_quantum_tensor(quantum_tensor)
        classical_tensors.append(tensor)

    tn = qtn.TensorNetwork(classical_tensors)
    return tn.contract(all, optimize="auto")
