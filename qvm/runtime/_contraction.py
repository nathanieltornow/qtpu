import quimb.tensor as qtn
import cotengra as ctg

from qvm.tensor import HybridTensorNetwork
from qvm.runtime.qpu_manager import QPUManager


def evaluate_quantum_tensors(
    hybrid_tn: HybridTensorNetwork, qpu_manager: QPUManager
) -> qtn.TensorNetwork:
    quantum_tensors = hybrid_tn.quantum_tensors
    tensors = [qpu_manager.run_quantum_tensor(qtens) for qtens in quantum_tensors]
    return qtn.TensorNetwork(tensors + hybrid_tn.classical_tensors)
