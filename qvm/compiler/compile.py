import optuna
from qiskit.circuit import QuantumCircuit


from qvm.ir import HybridCircuitIR
from qvm.tensor import HybridTensorNetwork
from .partition_optimizer import TreeOptimizer
from .oracle import LeafOracle

from .util import get_leafs


def compile(
    circuit: QuantumCircuit,
    oracle: LeafOracle | None = None,
    compression_methods: list[str] | None = None,
    max_trials: int = 100,
    show_progress_bar: bool = False,
) -> HybridTensorNetwork:
    if compression_methods is None:
        compression_methods = ["qubits", "2q", "none"]

    if oracle is None:
        oracle = LeafOracle()

    ir = HybridCircuitIR(circuit)

    opt = TreeOptimizer(oracle)

    return hyper_optimize(
        ir,
        opt,
        compression_methods=compression_methods,
        n_trials=max_trials,
        show_progress_bar=show_progress_bar,
    )


def _objective(
    trial: optuna.Trial,
    ir: HybridCircuitIR,
    opt: TreeOptimizer,
    compression_methods: list[str],
) -> float:

    compress = trial.suggest_categorical("compress", compression_methods)
    random_strength = trial.suggest_float("random_strength", 0.01, 10.0)
    weight_edges = trial.suggest_categorical("weight_edges", ["const", "log"])
    imbalance = trial.suggest_float("imbalance", 0.01, 1.0)
    imbalance_decay = trial.suggest_float("imbalance_decay", -5, 5)
    parts = trial.suggest_int("parts", 2, 16)
    parts_decay = trial.suggest_float("parts_decay", 0.0, 1.0)
    mode = trial.suggest_categorical("mode", ["direct", "recursive"])
    objective = trial.suggest_categorical("objective", ["cut", "km1"])
    fix_output_nodes = trial.suggest_categorical("fix_output_nodes", ["auto", ""])

    ir, tree = opt.optimize(
        ir,
        compress=compress,
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

    return tree.contraction_cost()


def hyper_optimize(
    ir: HybridCircuitIR,
    optimizer: TreeOptimizer,
    compression_methods: list[str],
    n_trials: int = 100,
    show_progress_bar: bool = False,
) -> HybridTensorNetwork:
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: _objective(trial, ir, optimizer, compression_methods),
        n_trials=n_trials,
        show_progress_bar=show_progress_bar,
    )

    ir = study.best_trial.user_attrs["ir"]
    return ir.hybrid_tn(
        [set(leaf) for leaf in get_leafs(study.best_trial.user_attrs["tree"])]
    )
