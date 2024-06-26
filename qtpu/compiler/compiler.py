from typing import Callable

import optuna
import cotengra as ctg
import numpy as np
from qiskit.circuit import QuantumCircuit, Barrier

from qtpu.ir import HybridCircuitIR
from qtpu.tensor import HybridTensorNetwork

from qtpu.compiler.optimizer import optimize
from qtpu.compiler.compress import CompressedIR
from qtpu.compiler.util import get_leafs
from qtpu.compiler.success import success_probability_static

LOGGING = False

if not LOGGING:
    optuna.logging.set_verbosity(optuna.logging.WARNING)


def compile_circuit(
    circuit: QuantumCircuit,
    success_fn: Callable[[CompressedIR, ctg.ContractionTree], float] | None = None,
    terminate_fn: Callable[[CompressedIR, ctg.ContractionTree], bool] | None = None,
    max_cost: int | tuple[int, int] | list[int] = np.inf,
    choose_leaf_methods: list[str] | None = None,
    compression_methods: list[str] | None = None,
    # function to choos the value from the pareto front
    n_trials: int = 100,
    show_progress_bar: bool = False,
) -> HybridTensorNetwork:

    study = hyper_optimize(
        circuit,
        success_fn=success_fn,
        terminate_fn=terminate_fn,
        max_cost=max_cost,
        choose_leaf_methods=choose_leaf_methods,
        compression_methods=compression_methods,
        n_trials=n_trials,
        show_progress_bar=show_progress_bar,
    )

    # best_trial = max(study.best_trials, key=lambda trial: pareto_fn(*trial.values))
    best_trial = find_best_trial(study)

    return trial_to_hybrid_tn(best_trial)


def trial_to_hybrid_tn(trial: optuna.Trial) -> HybridTensorNetwork:
    return trial.user_attrs["ir"].hybrid_tn(list(get_leafs(trial.user_attrs["tree"])))


def hyper_optimize(
    circuit: QuantumCircuit,
    # hyperargs
    success_fn: Callable[[CompressedIR, ctg.ContractionTree], float] | None = None,
    terminate_fn: Callable[[CompressedIR, ctg.ContractionTree], bool] | None = None,
    max_cost: int | tuple[int, int] | list[int] = np.inf,
    choose_leaf_methods: list[str] | None = None,
    compression_methods: list[str] | None = None,
    # optuna args
    n_trials: int = 100,
    show_progress_bar: bool = False,
) -> optuna.Study:

    if success_fn is None:
        success_fn = success_probability_static

    if compression_methods is None:
        compression_methods = ["qubits", "2q", "none"]

    if choose_leaf_methods is None:
        choose_leaf_methods = ["qubits", "nodes", "random"]

    ir = HybridCircuitIR(_remove_barriers(circuit))

    study = optuna.create_study(directions=["minimize", "maximize"])
    study.optimize(
        lambda trial: objective(
            trial,
            ir=ir,
            success_fn=success_fn,
            terminate_fn=terminate_fn,
            max_cost=max_cost,
            choose_leaf_methods=choose_leaf_methods,
            compression_methods=compression_methods,
        ),
        n_trials=n_trials,
        show_progress_bar=show_progress_bar,
    )
    return study


def objective(
    trial: optuna.Trial,
    ir: HybridCircuitIR,
    success_fn: Callable[[CompressedIR, ctg.ContractionTree], float],
    terminate_fn: Callable[[CompressedIR, ctg.ContractionTree], bool] | None = None,
    max_cost: int | tuple[int, int] | list[int] = np.inf,
    choose_leaf_methods: list[str] | None = None,
    compression_methods: list[str] | None = None,
) -> float:

    if isinstance(max_cost, tuple):
        assert max_cost[0] < max_cost[1]
        max_cost = trial.suggest_int("max_cost", *max_cost)
    elif isinstance(max_cost, list):
        max_cost = trial.suggest_categorical("max_cost", max_cost)

    compress = trial.suggest_categorical("compress", compression_methods)
    choose_leaf = trial.suggest_categorical("choose_leaf", choose_leaf_methods)

    # partition arguments
    random_strength = trial.suggest_float("random_strength", 0.01, 10.0)
    weight_edges = trial.suggest_categorical("weight_edges", ["const", "log"])
    imbalance = trial.suggest_float("imbalance", 0.01, 1.0)
    imbalance_decay = trial.suggest_float("imbalance_decay", -5, 5)
    parts = trial.suggest_int("parts", 2, 16)
    parts_decay = trial.suggest_float("parts_decay", 0.0, 1.0)
    mode = trial.suggest_categorical("mode", ["direct", "recursive"])
    objective = trial.suggest_categorical("objective", ["cut", "km1"])
    fix_output_nodes = trial.suggest_categorical("fix_output_nodes", ["auto", ""])

    ir, tree = optimize(
        ir,
        # optimizer arguments
        terminate_fn=terminate_fn,
        max_cost=max_cost,
        choose_leaf=choose_leaf,
        compress=compress,
        # partition arguments
        random_strength=random_strength,
        weight_edges=weight_edges,
        imbalance=imbalance,
        imbalance_decay=imbalance_decay,
        parts=parts,
        parts_decay=parts_decay,
        mode=mode,
        objective=objective,
        fix_output_nodes=fix_output_nodes,
    )

    trial.set_user_attr("ir", ir)
    trial.set_user_attr("tree", tree)

    if terminate_fn is not None and not terminate_fn(ir, tree):
        # optimization returned due to max_cost,
        # but termination condition was not met
        # make sure that the trial is not considered
        return np.inf, 0.0

    return tree.contraction_cost(), success_fn(ir, tree)


def _remove_barriers(circuit: QuantumCircuit) -> QuantumCircuit:
    new_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    for instr in circuit:
        if isinstance(instr.operation, Barrier):
            continue
        new_circuit.append(instr)
    return new_circuit


def find_best_trial(
    study: optuna.Study,
) -> optuna.Trial:
    best_trials = study.best_trials

    costs = np.array([trial.values[0] for trial in best_trials])
    success = np.array([trial.values[1] for trial in best_trials])

    if costs.max() == np.inf:
        raise ValueError(
            "No trial with finite cost found. "
            + "This is likely due to the termination condition not being met for any trial."
        )

    norm_costs = (costs - costs.min()) / (costs.max() - costs.min() + 1e-9)
    norm_success = (success - success.min()) / (success.max() - success.min() + 1e-9)

    norm_points = np.stack([norm_costs, norm_success], axis=1)

    ideal_point = np.array([0, 1])

    distances = np.linalg.norm(norm_points - ideal_point, axis=1)
    best_index = np.argmin(distances)

    return best_trials[best_index]
