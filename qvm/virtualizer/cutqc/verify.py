import pickle
import argparse
import glob
import numpy as np
from qiskit.quantum_info import Statevector

# from qiskit_helper_functions.non_ibmq_functions import evaluate_circ
# from qiskit_helper_functions.conversions import quasi_to_real
# from qiskit_helper_functions.metrics import chi2_distance, MSE, MAPE, cross_entropy, HOP


def verify(full_circuit, unordered, complete_path_map, subcircuits, smart_order):
    from qiskit_aer import StatevectorSimulator

    ground_truth = StatevectorSimulator().run(full_circuit).result().get_statevector()
    ground_truth = Statevector(ground_truth).probabilities()

    print(ground_truth)
    print(unordered)

    metrics = {}
    for quasi_conversion_mode in ["nearest", "naive"]:
        real_probability = quasi_to_real(
            quasiprobability=unordered, mode=quasi_conversion_mode
        )

        chi2 = chi2_distance(target=ground_truth, obs=real_probability)
        # mse = MSE(target=ground_truth, obs=real_probability)
        # mape = MAPE(target=ground_truth, obs=real_probability)
        # ce = cross_entropy(target=ground_truth, obs=real_probability)
        # hop = HOP(target=ground_truth, obs=real_probability)
        metrics[quasi_conversion_mode] = {
            "chi2": chi2,
            # "Mean Squared Error": mse,
            # "Mean Absolute Percentage Error": mape,
            # "Cross Entropy": ce,
            # "HOP": hop,
        }
    return unordered, metrics


def nearest_probability_distribution(quasiprobability):
    """Takes a quasiprobability distribution and maps
    it to the closest probability distribution as defined by
    the L2-norm.
    Parameters:
        return_distance (bool): Return the L2 distance between distributions.
    Returns:
        ProbDistribution: Nearest probability distribution.
        float: Euclidean (L2) distance of distributions.
    Notes:
        Method from Smolin et al., Phys. Rev. Lett. 108, 070502 (2012).
    """
    sorted_probs, states = zip(
        *sorted(zip(quasiprobability, range(len(quasiprobability))))
    )
    num_elems = len(sorted_probs)
    new_probs = np.zeros(num_elems)
    beta = 0
    diff = 0
    for state, prob in zip(states, sorted_probs):
        temp = prob + beta / num_elems
        if temp < 0:
            beta += prob
            num_elems -= 1
            diff += prob * prob
        else:
            diff += (beta / num_elems) * (beta / num_elems)
            new_probs[state] = prob + beta / num_elems
    return new_probs


def quasi_to_real(quasiprobability, mode):
    """
    Convert a quasi probability to a valid probability distribution
    """
    if mode == "nearest":
        return nearest_probability_distribution(quasiprobability=quasiprobability)
    elif mode == "naive":
        return naive_probability_distribution(quasiprobability=quasiprobability)
    else:
        raise NotImplementedError("%s conversion is not implemented" % mode)


def chi2_distance(target, obs):
    import copy

    target = copy.deepcopy(target)
    obs = copy.deepcopy(obs)
    obs = np.absolute(obs)
    if isinstance(target, np.ndarray):
        assert len(target) == len(obs)
        distance = 0
        for t, o in zip(target, obs):
            if abs(t - o) > 1e-10:
                distance += np.power(t - o, 2) / (t + o)
    elif isinstance(target, dict):
        distance = 0
        for o_idx, o in enumerate(obs):
            if o_idx in target:
                t = target[o_idx]
                if abs(t - o) > 1e-10:
                    distance += np.power(t - o, 2) / (t + o)
            else:
                distance += o
    else:
        raise Exception("Illegal target type:", type(target))
    return distance

def naive_probability_distribution(quasiprobability):
    """
    Takes a quasiprobability distribution and does the following two steps:
    1. Update all negative probabilities to 0
    2. Normalize
    """
    new_probs = np.where(quasiprobability < 0, 0, quasiprobability)
    new_probs /= np.sum(new_probs)
    return new_probs