"""Main entrypoint module for contracting hybrid tensor networks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import quimb.tensor as qtn

from qtpu.evaluators._estimator import ExpvalEvaluator
from qtpu.helpers import nearest_probability_distribution
from qtpu.transforms import circuit_to_hybrid_tn, wire_cuts_to_moves

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit

    from qtpu.evaluators._evaluator import CircuitTensorEvaluator
    from qtpu.tensor import HybridTensorNetwork


def evaluate(
    hybrid_tn: HybridTensorNetwork,
    evaluator: CircuitTensorEvaluator | None = None,
) -> qtn.TensorNetwork:
    """Evaluate the quantum tesnors of a hybrid tensor network using a specified evaluator.

    Parameters:
        hybrid_tn (HybridTensorNetwork): The hybrid tensor network to be evaluated.
        evaluator (CircuitTensorEvaluator | None, optional): The evaluator to use for
            evaluating the quantum tensors. If None, a default ExpvalEvaluator is used.

    Returns:
        qtn.TensorNetwork: The resulting classical tensor network after evaluation.
    """
    if evaluator is None:
        evaluator = ExpvalEvaluator()

    eval_tensors = evaluator.evaluate_batch(hybrid_tn.qtensors)
    return qtn.TensorNetwork(eval_tensors + hybrid_tn.ctensors)


def contract(
    hybrid_tn: HybridTensorNetwork,
    evaluator: CircuitTensorEvaluator | None = None,
) -> qtn.Tensor:
    """Contract the hybrid tensor network.

    Parameters:
        hybrid_tn (HybridTensorNetwork): The hybrid tensor network to be contracted.
        evaluator (CircuitTensorEvaluator | None, optional): The evaluator to use for
            evaluating the quantum tensors. If None, a default ExpvalEvaluator is used.

    Returns:
        qtn.Tensor: The resulting tensor after contraction.
    """
    tn = evaluate(hybrid_tn, evaluator)
    return tn.contract(all, optimize="auto-hq", output_inds=[])


def execute(
    circuit: QuantumCircuit, evaluator: CircuitTensorEvaluator | None = None
) -> qtn.Tensor | float:
    """Execute a quantum circuit using hybrid tensor network contraction.

    Parameters:
        circuit (QuantumCircuit): The quantum circuit to be executed.
        evaluator (CircuitTensorEvaluator | None, optional): The evaluator to use for
            evaluating the quantum tensors. If None, a default ExpvalEvaluator is used.

    Returns:
        qtn.Tensor | float: The result of the circuit execution.
    """
    circuit = wire_cuts_to_moves(circuit)
    hybrid_tn = circuit_to_hybrid_tn(circuit)
    return contract(hybrid_tn, evaluator)


def sample(tn: qtn.TensorNetwork, num_samples: int = 10) -> list[str]:
    """Sample from a tensor network.

    Assumes that the tensor network is a tensor network representing a probability distribution.
    Each outer index of the tensor network is assumed to represent a qubit measurment outcome.

    Parameters:
        tn (qtn.TensorNetwork): The tensor network to sample from.
        num_samples (int, optional): The number of samples to generate.

    Returns:
        list[str]: A list of binary strings representing the samples (Qiskit convention).
    """
    outer_inds = sorted(tn.outer_inds())
    assert all(tn.ind_size(ind) == 2 for ind in outer_inds)

    rng = np.random.default_rng()

    outputs = []
    for _ in range(num_samples):
        output = ""
        tn_ = tn.copy()
        for ind in outer_inds:
            result = tn_.contract(all, output_inds=[ind]).data

            result /= sum(result)

            result = nearest_probability_distribution(result)
            result = np.array([result.get(0, 0), result.get(1, 0)])

            sample = rng.choice(2, p=result)

            arr = np.array([1, 0]) if sample == 0 else np.array([0, 1])

            tn_.add_tensor(qtn.Tensor(arr, inds=[ind]))

            output = str(sample) + output

        outputs.append(output)

    return outputs


def get_quasi_probability(tn: qtn.TensorNetwork, bits: int | str) -> float:
    """Get the quasi-probability of a specific bitstring from a tensor network.

    Parameters:
        tn (qtn.TensorNetwork): The tensor network to calculate the quasi-probability from.
        bits (int | str): The bitstring to calculate the quasi-probability for.

    Returns:
        float: The quasi-probability of the specified bitstring.
    """
    if isinstance(bits, int):
        bits = f"{bits:0{tn.num_outer_inds()}b}"

    outer_inds = sorted(tn.outer_inds())
    assert all(
        tn.ind_size(ind) == 2 for ind in outer_inds
    ), "Outer indices must be qubits-measurement outcomes."

    assert len(bits) == tn.num_outer_inds(), "Bitstring must match number of qubits."

    tn_ = tn.copy()
    for ind, bit in zip(outer_inds, reversed(bits), strict=False):
        arr = np.array([1, 0]) if bit == "0" else np.array([0, 1])
        tn_.add_tensor(qtn.Tensor(arr, inds=[ind]))

    assert len(tn_.outer_inds()) == 0, "Tensor network must be fully contracted."

    return float(tn_.contract(all))
