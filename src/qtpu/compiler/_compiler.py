from __future__ import annotations

from typing import TYPE_CHECKING

from qtpu.compiler._opt import hyper_optimize

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


def cut(
    circuit: QuantumCircuit,
    max_sampling_cost: float = 1e6,
    # optuna args
    num_threads: int = 1,
    n_trials: int = 100,
    show_progress_bar: bool = False,
) -> QuantumCircuit:
    return hyper_optimize(
        circuit=circuit,
        max_sampling_cost=max_sampling_cost,
        num_threads=num_threads,
        n_trials=n_trials,
        show_progress_bar=show_progress_bar,
    )
