import optuna
from qiskit.circuit import QuantumCircuit


from qvm.ir import HybridCircuitIR, HybridCircuitIface
from qvm.tensor import HybridTensorNetwork
from .partition_optimizer import Optimizer
from .oracle import Oracle
from .compression import compress_qubits
from .util import traverse, get_leafs


optuna.logging.set_verbosity(optuna.logging.WARNING)


def compile(
    circuit: QuantumCircuit,
    compress: str | None = None,
    oracle: Oracle | None = None,
    max_cost: int = 100000,
    max_trials: int = 100,
    show_progress_bar: bool = False,
) -> HybridTensorNetwork:

    if oracle is None:
        oracle = Oracle()

    ir = HybridCircuitIR(circuit)
    # frontend
    match compress:
        case "qubit":
            ir = compress_qubits(ir)

    opt = Optimizer(oracle, max_cost)
    return hyper_optimize(opt, ir, max_trials, show_progress_bar)


def _objective(trial: optuna.Trial, opt: Optimizer, ir: HybridCircuitIface) -> float:
    random_strength = trial.suggest_float("random_strength", 0.01, 10.0)
    weight_edges = trial.suggest_categorical("weight_edges", ["const", "log"])
    imbalance = trial.suggest_float("imbalance", 0.01, 1.0)
    imbalance_decay = trial.suggest_float("imbalance_decay", -5, 5)
    parts = trial.suggest_int("parts", 2, 16)
    parts_decay = trial.suggest_float("parts_decay", 0.0, 1.0)
    mode = trial.suggest_categorical("mode", ["direct", "recursive"])
    objective = trial.suggest_categorical("objective", ["cut", "km1"])
    fix_output_nodes = trial.suggest_categorical("fix_output_nodes", ["auto", ""])

    tree = opt.optimize(
        ir,
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

    trial.set_user_attr("tree", tree)
    return tree.contraction_cost()


def hyper_optimize(
    optimizer: Optimizer,
    ir: HybridCircuitIR,
    n_trials: int = 100,
    show_progress_bar: bool = False,
) -> HybridTensorNetwork:
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: _objective(trial, optimizer, ir),
        n_trials=n_trials,
        show_progress_bar=show_progress_bar,
    )
    best_tree = study.best_trial.user_attrs["tree"]
    return ir.hybrid_tn([set(leaf) for leaf in get_leafs(best_tree)])
