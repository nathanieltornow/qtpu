import abc

import quimb.tensor as qtn

from qtpu.tensor import CircuitTensor


class CircuitTensorEvaluator(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, circuit_tensor: "CircuitTensor") -> qtn.Tensor:
        pass

    def evaluate_batch(
        self, circuit_tensors: list["CircuitTensor"]
    ) -> list[qtn.Tensor]:
        return [self.evaluate(circuit_tensor) for circuit_tensor in circuit_tensors]
