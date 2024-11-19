import numpy as np
import quimb.tensor as qtn
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import BaseEstimatorV2
from qiskit_aer.primitives import EstimatorV2


from qtpu.tensor import CircuitTensor
from qtpu.transforms import decompose_qpd_measures, remove_operations_by_name
from .evaluator import CircuitTensorEvaluator


class SimExpvalEvaluator(CircuitTensorEvaluator):
    def __init__(self, estimator: BaseEstimatorV2 | None = None):
        if estimator is None:
            estimator = EstimatorV2()
        assert isinstance(estimator, BaseEstimatorV2)

        self.estimator = estimator

    def evaluate(self, circuit_tensor: CircuitTensor) -> qtn.Tensor:
        circuits = circuit_tensor.flat()
        circuits = [c.decompose() for c in circuits]
        circuits = [
            decompose_qpd_measures(c, defer=True, inplace=True).decompose()
            for c in circuits
        ]

        observables = [_get_Z_observable(c) for c in circuits]

        for c in circuits:
            c.remove_final_measurements()
            remove_operations_by_name(c, {"reset"})
            assert not any(instr.operation.name in {"measure", "reset"} for instr in c)

        results = self.estimator.run(
            [(c, o) for c, o in zip(circuits, observables)]
        ).result()
        expvals = [res.data.evs for res in results]
        return qtn.Tensor(
            np.array(expvals).reshape(circuit_tensor.shape), inds=circuit_tensor.inds
        )


def _get_Z_observable(circuit: QuantumCircuit) -> str:
    measured_qubits = _get_meas_qubits(circuit)
    obs = ["I"] * circuit.num_qubits
    for qubit in measured_qubits:
        obs[qubit] = "Z"
    return "".join(reversed(obs))


def _get_meas_qubits(circuit: QuantumCircuit) -> list[int]:
    measured_qubits = sorted(
        set(
            circuit.qubits.index(instr.qubits[0])
            for instr in circuit
            if instr.operation.name == "measure"
        ),
    )
    return measured_qubits
