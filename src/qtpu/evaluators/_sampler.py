from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import numpy as np
import quimb.tensor as qtn
from qiskit.primitives import BaseSamplerV2, BitArray
from qiskit_aer.primitives import SamplerV2

from qtpu.transforms import decompose_qpd_measures

from ._evaluator import CircuitTensorEvaluator

if TYPE_CHECKING:
    from qtpu.tensor import CircuitTensor


class SamplerEvaluator(CircuitTensorEvaluator):
    """Evaluator for computing sampling probabilities of quantum tensors.

    Attributes:
    ----------
    sampler : BaseSamplerV2
        The Qiskit sampler to use for evaluating the sampling probabilities.
    """

    def __init__(
        self, sampler: BaseSamplerV2 | None = None, shots: int = 20000
    ) -> None:
        """Initialize the evaluator.

        Parameters:
            sampler (BaseSamplerV2 | None, optional): The sampler to use for
                evaluating the sampling probabilities. If None, a default Aer's SamplerV2 is used.
            shots (int, optional): The number of shots to use for sampling.
        """
        if sampler is None:
            sampler = SamplerV2()
        assert isinstance(sampler, BaseSamplerV2)

        self.sampler = sampler
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

        results = self.sampler.run(circuits, shots=self.shots).result()

        arrays = []

        for result in results:
            data_dict = {key: result.data[key] for key in result.data}

            qpd_measures = data_dict.pop("qpd_measurements", None)

            data_tupels = sorted(
                data_dict.items(),
                key=operator.itemgetter(0),
            )

            keys, bitarrays = zip(*data_tupels, strict=False)

            merged_array = BitArray.concatenate_bits(list(reversed(bitarrays)))

            prob_array = np.zeros(shape=(2 ** len(keys)), dtype=np.float32)

            num_shots = merged_array.num_shots

            if qpd_measures is not None:

                prob_array1 = np.zeros(shape=(2 ** len(keys)), dtype=np.float32)

                for res1, qpd_bitcnt in zip(
                    merged_array.array, qpd_measures.bitcount(), strict=False
                ):
                    if qpd_bitcnt % 2 == 0:
                        prob_array[int.from_bytes(res1.tobytes(), "big")] += 1
                    else:
                        prob_array1[int.from_bytes(res1.tobytes(), "big")] -= 1

                prob_array = (prob_array + prob_array1) / num_shots
            else:
                for res1 in merged_array.array:
                    prob_array[int.from_bytes(res1.tobytes(), "big")] += 1

                prob_array /= num_shots

            prob_array = prob_array.reshape((2,) * len(keys))

            arrays.append(prob_array)

        full_result = np.array(arrays).reshape(circuit_tensor.shape + prob_array.shape)

        return qtn.Tensor(full_result, inds=circuit_tensor.inds + tuple(keys))
