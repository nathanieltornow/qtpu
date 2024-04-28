from quimb.tensor import Tensor
from cotengra import ContractionTree

from qvm.tensor import QuantumTensor, QuantumTensorNetwork


def evaluate_quantum_tensor(quantum_tensor, qpu_manager_client) -> Tensor:
    # 1. evaluate the circuits to a tensor
    # 2. do post-processing on the tensor for every FIRST gate, special case for wires

    pass


def evaluate_tensor_network(
    qtn: QuantumTensorNetwork, contraction_tree: ContractionTree, qpu_manager_client
) -> float:
    pass


def create_dummy_tensor_network(qtn: QuantumTensorNetwork) -> Tensor:
    pass



class CircuitKnitter:
    def __init__(self, virtual_gates) -> None:
        pass


