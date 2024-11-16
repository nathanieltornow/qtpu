from dataclasses import dataclass

import numpy as np
from qiskit.circuit import QuantumCircuit


@dataclass(frozen=True)
class CircuitVector:
    circuits: list[QuantumCircuit]
    ind: str | None = None
    weights: list[float] | None = None


class CircuitTensor:
    def __init__(self, vectors: list[CircuitVector]):
        self._vectors = vectors
        self._inds = tuple(v.ind for v in vectors if v.ind is not None)
        self._shape = tuple(len(v.circuits) for v in vectors if v.ind is None)

    @property
    def shape(self):
        return self._shape

    @property
    def inds(self):
        return self._inds

    def __getitem__(self, key: tuple[int, ...]) -> QuantumCircuit:
        circuit = QuantumCircuit()
        key_idx = 0
        for vector in self._vectors:
            if vector.ind is None:
                circuit = circuit.compose(vector.circuits[0])
            else:
                circuit = circuit.compose(vector.circuits[key[key_idx]])
                key_idx += 1
        return circuit

    def to_dense(self) -> np.ndarray:
        """
        Convert the tensor to a dense representation.

        Returns:
            np.ndarray: A dense representation of the tensor
                with QuantumCircuits as elements.
        """
        indices = np.ndindex(self.shape)
        tensor_list = [self[tuple(idx)] for idx in indices]
        return np.array(tensor_list, dtype=object).reshape(self.shape)
