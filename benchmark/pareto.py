import optuna
import matplotlib.pyplot as plt
import numpy as np


def plot_pareto_front(study: optuna.Study, ax: plt.Axes) -> None:
    trials = set(tuple(trial.values) for trial in study.trials)
    best_trials = sorted(
        set(tuple(trial.values) for trial in study.best_trials),
        key=lambda x: (x[0], -x[1]),
    )
    non_optimal_trials = trials - set(best_trials)

    print(f"Best trials: {best_trials}")

    print(
        np.gradient(
            [vals[1] for vals in best_trials], [vals[0] for vals in best_trials]
        )
    )

    ax.plot(
        [vals[0] for vals in best_trials],
        [vals[1] for vals in best_trials],
        "o-",
        label="Pareto optimal",
    )
    ax.plot(
        [vals[0] for vals in non_optimal_trials],
        [vals[1] for vals in non_optimal_trials],
        "x",
        label="Non-optimal",
    )

    ax.set_xlabel("Cost")
    ax.set_ylabel("Success")
