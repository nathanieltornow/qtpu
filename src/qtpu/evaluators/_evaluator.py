from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import quimb.tensor as qtn

    from qtpu.tensor import CircuitTensor


class CircuitTensorEvaluator(abc.ABC):
    """Abstract base class for evaluating circuit tensors."""

    @abc.abstractmethod
    def evaluate(self, circuit_tensor: CircuitTensor) -> qtn.Tensor:
        """Evaluate a single circuit tensor to a classical tensor.

        Args:
            circuit_tensor (CircuitTensor): The circuit tensor to evaluate.

        Returns:
            qtn.Tensor: The resulting classical tensor.
        """
        ...

    def evaluate_batch(
        self, circuit_tensors: list[CircuitTensor]
    ) -> list[qtn.Tensor]:
        """Evaluate a batch of circuit tensors to classical tensors.

        Args:
            circuit_tensors (list[CircuitTensor]): The circuit tensors to evaluate.

        Returns:
            list[qtn.Tensor]: The resulting classical tensors.
        """
        return [self.evaluate(circuit_tensor) for circuit_tensor in circuit_tensors]
