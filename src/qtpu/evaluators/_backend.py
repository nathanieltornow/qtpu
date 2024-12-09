from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import quimb.tensor as qtn
from qiskit.providers import BackendV2

from qtpu.transforms import decompose_qpd_measures, squash_cregs

from ._evaluator import CircuitTensorEvaluator

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit
    from qtpu.tensor import CircuitTensor
    from numpy.typing import NDArray


class BackendEvaluator(CircuitTensorEvaluator):
    """Evaluator for computing sampling probabilities of quantum tensors using qiskit backend.

    Attributes:
    ----------
    backend : BackendV2
        The Qiskit backend to use for evaluating the sampling probabilities.
    shots : int
        The number of shots to use for sampling.
    """

    def __init__(self, backend: BackendV2, shots: int = 20000) -> None:
        """Initialize the evaluator.

        Parameters:
            sampler (BaseSamplerV2 | None, optional): The sampler to use for
                evaluating the sampling probabilities. If None, a default Aer's SamplerV2 is used.
            shots (int, optional): The number of shots to use for sampling.
        """
        assert isinstance(backend, BackendV2)
        self.backend = backend
        self.shots = shots

    def evaluate(self, circuit_tensor: CircuitTensor) -> qtn.Tensor:
        """Evaluate a single circuit tensor to a classical tensor.

        Args:
            circuit_tensor (CircuitTensor): The circuit tensor to evaluate.

        Returns:
            qtn.Tensor: The resulting classical tensor with the sampling probabilities.
                The tensor has additional indices corresponding to the measured qubits.
        """
        circuits = circuit_tensor.flat()

        circuits = [c.decompose() for c in circuits]
        circuits = [
            decompose_qpd_measures(c, defer=True, inplace=True).decompose()
            for c in circuits
        ]

        num_result_bits = sum(
            1 for c in circuits[0].cregs if c.name != "qpd_measurements"
        )

        circuits = [squash_cregs(c) for c in circuits]

        def get_qpd_reg_len(circuit: QuantumCircuit) -> int:
            if circuit.cregs[-1].name == "qpd_measurements":
                return circuit.cregs[-1].size
            return 0

        def prepare_counts(counts: dict[str, int], num_bits: int) -> dict[str, int]:
            if num_bits <= 0:
                return counts
            new_counts = {}
            for key, value in counts.items():
                # get the most significant bits
                qpd_meas = key[:num_bits]
                factor = 1 if qpd_meas.count("1") % 2 == 0 else -1
                new_key = key[num_bits:]
                new_counts[new_key] = new_counts.get(new_key, 0) + factor * value
            return new_counts

        def counts_to_probtensor(counts: dict[str, int]) -> NDArray[np.float32]:
            num_shots = sum(counts.values())
            arr = np.ndarray(shape=(2**num_result_bits,))
            for key, value in counts.items():
                arr[int(key.replace(" ", "")[::-1], 2)] = float(value) / float(
                    num_shots
                )
            return arr.reshape((2,) * num_result_bits)

        counts = self.backend.run(circuits, shots=self.shots).result().get_counts()

        counts = [counts] if isinstance(counts, dict) else counts

        counts = [
            prepare_counts(count, get_qpd_reg_len(c))
            for c, count in zip(circuits, counts)
        ]

        prob_arrays = [counts_to_probtensor(count) for count in counts]
        shape = prob_arrays[0].shape

        full_result = np.array(prob_arrays).reshape(circuit_tensor.shape + shape)

        keys = [
            creg.name for creg in circuits[0].cregs if creg.name != "qpd_measurements"
        ]

        return qtn.Tensor(full_result, inds=circuit_tensor.inds + tuple(keys))
