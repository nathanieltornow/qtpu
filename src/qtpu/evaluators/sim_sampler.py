import numpy as np
import quimb.tensor as qtn
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import DataBin, BitArray
from qiskit.primitives import BaseSamplerV2
from qiskit_aer.primitives import SamplerV2

from qtpu.tensor import CircuitTensor
from qtpu.transforms import decompose_qpd_measures, remove_operations_by_name
from .evaluator import CircuitTensorEvaluator


class SamplerEvaluator(CircuitTensorEvaluator):
    def __init__(self, sampler: BaseSamplerV2 | None = None, shots: int = 20000):
        if sampler is None:
            sampler = SamplerV2()
        assert isinstance(sampler, BaseSamplerV2)

        self.sampler = sampler
        self.shots = shots

    def evaluate(self, circuit_tensor: CircuitTensor) -> qtn.Tensor:
        circuits = circuit_tensor.flat()

        circuits = [c.decompose() for c in circuits]
        circuits = [
            decompose_qpd_measures(c, defer=True, inplace=True).decompose()
            for c in circuits
        ]

        results = self.sampler.run(circuits, shots=self.shots).result()

        arrays = []

        for result in results:
            data_dict = {key: result.data[key] for key in result.data.keys()}

            qpd_measures = data_dict.pop("qpd_measurements", None)

            data_tupels = sorted(
                [(key, bitarray) for key, bitarray in data_dict.items()],
                key=lambda x: x[0],
            )

            keys, bitarrays = zip(*data_tupels)

            merged_array = BitArray.concatenate_bits(list(reversed(bitarrays)))

            prob_array = np.zeros(shape=(2 ** len(keys)), dtype=np.float32)

            num_shots = merged_array.num_shots

            if qpd_measures is not None:

                prob_array1 = np.zeros(shape=(2 ** len(keys)), dtype=np.float32)

                for res1, qpd_bitcnt in zip(
                    merged_array.array, qpd_measures.bitcount()
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
