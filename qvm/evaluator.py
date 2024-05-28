import abc
from enum import Enum
from typing import Any

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV2
from qiskit import transpile


from qvm.tensor import QuantumTensor, ClassicalTensor
from qvm._quasi_distr import prepare_samples


class ResultType(Enum):
    SAMPLES = 0
    EXPECTATION = 1


class QTensorEvaluator(abc.ABC):
    @abc.abstractmethod
    def evaluate_qtensor(
        self, qtensor: QuantumTensor, shots: int
    ) -> ClassicalTensor: ...


class BackendEvaluator(QTensorEvaluator):
    def __init__(
        self,
        backend: BackendV2,
        result_type: ResultType = ResultType.SAMPLES,
        optimization_level: int = 3,
    ):
        self._backend = backend
        self._result_type = result_type
        self._optimization_level = optimization_level

    def evaluate_qtensor(self, qtensor: QuantumTensor, shots: int) -> ClassicalTensor:
        num_bits = qtensor.circuit.cregs[0].size

        circuits = [
            transpile(circ, self._backend, optimization_level=self._optimization_level)
            for circ, _ in qtensor.instances()
        ]
        shot_per_circ = [int(shots * weight) for _, weight in qtensor.instances()]

        cid_withour_meas = [
            (i, s)
            for i, (circ, s) in enumerate(zip(circuits, shot_per_circ))
            if circ.count_ops().get("measure", 0) == 0
        ]

        for i, _ in reversed(cid_withour_meas):
            circuits.pop(i)
            shot_per_circ.pop(i)

        jobs = [
            self._backend.run(circ, shots=s) for circ, s in zip(circuits, shot_per_circ)
        ]
        counts = [job.result().get_counts() for job in jobs]

        for i, s in cid_withour_meas:
            print(i, s)
            counts.insert(i, {"0": s})

        match self._result_type:
            case ResultType.SAMPLES:
                arr = np.array(
                    [prepare_samples(count, num_bits) for count in counts]
                ).reshape(qtensor.shape)
            case ResultType.EXPECTATION:
                arr = np.array([expval_from_counts(count) for count in counts]).reshape(
                    qtensor.shape
                )
            case _:
                raise ValueError(f"Invalid result type: {self._result_type}")

        return ClassicalTensor(arr, qtensor.inds)


class DummyEvaluator(QTensorEvaluator):
    def evaluate_qtensor(self, qtensor: QuantumTensor, _: int) -> ClassicalTensor:
        return ClassicalTensor(np.random.randn(qtensor.shape), qtensor.inds)


def expval_from_counts(counts: dict[str, int]) -> float:
    expval = 0.0
    shots = sum(counts.values())
    for bitstring, count in counts.items():
        parity = 1 - 2 * int(bitstring.count("1") % 2)
        expval += parity * (count / shots)
    return expval
