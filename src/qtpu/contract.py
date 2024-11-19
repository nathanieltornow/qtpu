import numpy as np
import quimb.tensor as qtn
from qiskit.circuit import QuantumCircuit

from qtpu.transforms import circuit_to_hybrid_tn, wire_cuts_to_moves
from qtpu.tensor import HybridTensorNetwork
from qtpu.evaluators.evaluator import CircuitTensorEvaluator
from qtpu.evaluators.sim_estimator import SimExpvalEvaluator
from qtpu.helpers import nearest_probability_distribution


def evaluate(
    hybrid_tn: HybridTensorNetwork,
    evaluator: CircuitTensorEvaluator | None = None,
) -> qtn.TensorNetwork:
    if evaluator is None:
        evaluator = SimExpvalEvaluator()

    eval_tensors = evaluator.evaluate_batch(hybrid_tn.qtensors)
    return qtn.TensorNetwork(eval_tensors + hybrid_tn.ctensors)


def contract(
    hybrid_tn: HybridTensorNetwork,
    evaluator: CircuitTensorEvaluator | None = None,
) -> qtn.Tensor:
    tn = evaluate(hybrid_tn, evaluator)
    return tn.contract(all, optimize="auto-hq", output_inds=[])


def execute(
    circuit: QuantumCircuit, evaluator: CircuitTensorEvaluator | None = None
) -> qtn.Tensor:
    circuit = wire_cuts_to_moves(circuit)
    hybrid_tn = circuit_to_hybrid_tn(circuit)
    return contract(hybrid_tn, evaluator)


def sample(tn: qtn.TensorNetwork, num_samples: int = 10) -> list[str]:
    outer_inds = sorted(tn.outer_inds())
    assert all(tn.ind_size(ind) == 2 for ind in outer_inds)

    outputs = []
    for _ in range(num_samples):
        output = ""
        tn_ = tn.copy()
        for ind in outer_inds:
            result = tn_.contract(all, output_inds=[ind]).data

            result /= sum(result)

            result = nearest_probability_distribution(result)
            result = np.array([result.get(0, 0), result.get(1, 0)])

            sample = np.random.choice(2, p=result)

            arr = np.array([1, 0]) if sample == 0 else np.array([0, 1])

            tn_.add_tensor(qtn.Tensor(arr, inds=[ind]))

            output = str(sample) + output

        outputs.append(output)

    return outputs
