from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from qtpu.compiler._opt import hyper_optimize
from qtpu.compiler._terminators import reach_num_qubits
from qtpu.transforms import circuit_to_hybrid_tn

if TYPE_CHECKING:

    import optuna
    from optuna.trial import FrozenTrial
    from qiskit.circuit import QuantumCircuit


def find_best_trial(
    study: optuna.Study,
) -> FrozenTrial:
    best_trials = study.best_trials

    costs = np.array([trial.values[0] for trial in best_trials])
    error = np.array([trial.values[1] for trial in best_trials])

    if costs.min() == np.inf:
        msg = (
            "No trial with finite cost found. "
            "This is likely due to the termination condition not being met for any trial."
        )
        raise ValueError(msg)

    norm_costs = (costs - costs.min()) / (costs.max() - costs.min() + 1e-9)
    norm_success = (error - error.min()) / (error.max() - error.min() + 1e-9)

    norm_points = np.stack([norm_costs, norm_success], axis=1)

    ideal_point = np.array([0, 0])

    distances = np.linalg.norm(norm_points - ideal_point, axis=1)
    best_index = np.argmin(distances)

    return best_trials[best_index]


def find_min_cost_trial(
    study: optuna.Study,
) -> FrozenTrial:
    best_trials = study.best_trials

    costs = np.array([trial.values[0] for trial in best_trials])

    if costs.min() == np.inf:
        msg = (
            "No trial with finite cost found. "
            "This is likely due to the termination condition not being met for any trial."
        )
        raise ValueError(msg)

    best_index = np.argmin(costs)
    return best_trials[best_index]


def cut(
    circuit: QuantumCircuit,
    num_qubits: int,
    max_overhead: float | tuple[float, float] | list[float] = np.inf,
    choose_leaf_methods: list[str] | None = None,
    compression_methods: list[str] | None = None,
    # optuna args
    n_trials: int = 100,
    show_progress_bar: bool = False,
) -> QuantumCircuit:
    study = hyper_optimize(
        circuit=circuit,
        error_fn=lambda circ: (
            0.0
            if any(
                c.num_qubits > num_qubits
                for c in circuit_to_hybrid_tn(circ).subcircuits
            )
            else 1.0
        ),
        max_overhead=max_overhead,
        terminate_fn=reach_num_qubits(num_qubits),
        choose_leaf_methods=choose_leaf_methods,
        compression_methods=compression_methods,
        n_trials=n_trials,
        show_progress_bar=show_progress_bar,
    )
    if max_overhead == np.inf:
        best_trial = find_min_cost_trial(study)
    else:
        best_trial = find_best_trial(study)

    return best_trial.user_attrs["circuit"]
