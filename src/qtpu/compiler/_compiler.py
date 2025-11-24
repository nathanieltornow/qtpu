from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from qtpu.compiler._opt import hyper_optimize
from qtpu.compiler._terminators import reach_num_qubits
from qtpu.transforms import circuit_to_hybrid_tn

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


def cut(
    circuit: QuantumCircuit,
    num_qubits: int,
    max_overhead: float | tuple[float, float] | list[float] = np.inf,
    # optuna args
    n_trials: int = 100,
    show_progress_bar: bool = False,
) -> QuantumCircuit:
    return hyper_optimize(
        circuit=circuit,
        max_overhead=max_overhead,
        terminate_fn=reach_num_qubits(num_qubits),
        n_trials=n_trials,
        show_progress_bar=show_progress_bar,
    )
