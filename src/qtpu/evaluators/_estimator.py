from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import quimb.tensor as qtn
from qiskit.primitives import BaseEstimatorV2
from qiskit_aer.primitives import EstimatorV2

from qtpu.transforms import decompose_qpd_measures, remove_operations_by_name

from ._evaluator import CircuitTensorEvaluator

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit

    from qtpu.tensor import CircuitTensor


class ExpvalEvaluator(CircuitTensorEvaluator):
    """Evaluator for computing <ZZ..Z> expectation values of quantum tensors.

    Attributes:
    ----------
    estimator : BaseEstimatorV2
        The Qiskit estimator to use for evaluating the expectation values.
    """

    def __init__(self, estimator: BaseEstimatorV2 | None = None) -> None:
        """Initialize the evaluator.

        Parameters:
            estimator (BaseEstimatorV2 | None, optional): The estimator to use for
                evaluating the expectation values. If None, a default Aer's EstimatorV2 is used.
        """
        if estimator is None:
            estimator = EstimatorV2()
        assert isinstance(estimator, BaseEstimatorV2)

        self.estimator = estimator

    def evaluate(self, circuit_tensor: CircuitTensor) -> qtn.Tensor:
        """Evaluate a single circuit tensor to a classical tensor.

        Args:
            circuit_tensor (CircuitTensor): The circuit tensor to evaluate.
        
        Returns:
            The resulting classical tensor with the expectation values 
                at respective indices.
        """
        circuits = circuit_tensor.flat()
        circuits = [c.decompose() for c in circuits]
        circuits = [
            decompose_qpd_measures(c, defer=True, inplace=True).decompose()
            for c in circuits
        ]

        observables = [_get_z_observable(c) for c in circuits]

        for c in circuits:
            c.remove_final_measurements()
            remove_operations_by_name(c, {"reset"})
            assert not any(instr.operation.name in {"measure", "reset"} for instr in c)

        results = self.estimator.run(
            list(zip(circuits, observables, strict=False))
        ).result()
        expvals = [res.data.evs for res in results]
        return qtn.Tensor(
            np.array(expvals).reshape(circuit_tensor.shape), inds=circuit_tensor.inds
        )


def _get_z_observable(circuit: QuantumCircuit) -> str:
    measured_qubits = _get_meas_qubits(circuit)
    obs = ["I"] * circuit.num_qubits
    for qubit in measured_qubits:
        obs[qubit] = "Z"
    return "".join(reversed(obs))


def _get_meas_qubits(circuit: QuantumCircuit) -> list[int]:
    return sorted(
        {
            circuit.qubits.index(instr.qubits[0])
            for instr in circuit
            if instr.operation.name == "measure"
        },
    )
