from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import cotengra as ctg
import numpy as np
import optuna

from qtpu.compiler._compress import CompressedIR, compress_2q_gates, compress_qubits
from qtpu.compiler._ir import HybridCircuitIR
from qtpu.compiler._util import get_leafs
from qtpu.transforms import remove_operations_by_name

if TYPE_CHECKING:
    from collections.abc import Callable
    from qiskit.circuit import QuantumCircuit

logger = logging.getLogger("qtpu.compiler")

if True:  # silence Optuna unless debugging
    optuna.logging.set_verbosity(optuna.logging.WARNING)


_TRIAL_RESULTS = {}


# =============================================================================
#  Quantum cost = worst-case decompressed node count of any leaf
# =============================================================================
def quantum_cost(ir: CompressedIR, tree: ctg.ContractionTree) -> float:
    leafs = get_leafs(tree)
    if not leafs:
        return 0.0
    return float(max(len(ir.decompress_nodes(leaf)) for leaf in leafs))


# =============================================================================
#  Leaf selection strategies
# =============================================================================
def max_nodes_leaf(ir: CompressedIR, tree: ctg.ContractionTree):
    if len(tree.childless) == 1:
        return cast(frozenset[int], next(iter(tree.childless)))
    if not tree.childless:
        return None
    return cast(
        frozenset[int],
        max(tree.childless, key=lambda x: len(ir.decompress_nodes(x))),
    )


def max_qubits_leaf(ir: CompressedIR, tree: ctg.ContractionTree):
    if len(tree.childless) == 1:
        return cast(frozenset[int], next(iter(tree.childless)))
    if not tree.childless:
        return None
    return cast(frozenset[int], max(tree.childless, key=ir.num_qubits))


# =============================================================================
#  Partition & contract helper
# =============================================================================
def partition_and_contract_subgraph(
    tree,
    tree_node,
    rand_size_dict,
    rng,
    parts,
    parts_decay,
    dynamic_imbalance,
    imbalance,
    imbalance_decay,
    dynamic_fix,
    partition_opts,
    super_optimize,
):
    """Perform one Kahypar-based bipartition and contract subgraphs."""
    subgraph = tuple(tree_node)
    subsize = len(subgraph)

    s = subsize / tree.N
    parts_s = max(int(parts * (s**parts_decay)), 2)

    if dynamic_imbalance:
        if imbalance_decay >= 0:
            partition_opts["imbalance"] = (s**imbalance_decay) * imbalance
        else:
            partition_opts["imbalance"] = 1 - s ** (-imbalance_decay) * (1 - imbalance)

    if dynamic_fix:
        parts_s = 2
        partition_opts["fix_output_nodes"] = s == 1.0

    inputs = tuple(map(tuple, tree.node_to_terms(subgraph)))
    output = tuple(tree.get_legs(tree_node))

    membership = ctg.pathfinders.path_kahypar.kahypar_subgraph_find_membership(
        inputs,
        output,
        rand_size_dict,
        parts=parts_s,
        seed=rng,
        **partition_opts,
    )

    new_subgs = tuple(
        map(ctg.core.node_from_seq, ctg.core.separate(subgraph, membership))
    )

    if len(new_subgs) > 1:
        tree.contract_nodes(new_subgs, optimize=super_optimize, check=None)


# =============================================================================
#  OPTIMIZE (with acceptance rule)
# =============================================================================
def optimize(
    circuit: QuantumCircuit,
    *,
    gamma_q: float = 1.10,  # require ≥10% quantum improvement
    gamma_c: float = 10.0,  # allow ≤10x classical blowup
    max_overhead: float = np.inf,
    choose_leaf: str = "nodes",
    compress: str = "none",
    random_strength: float = 0.01,
    parts: int = 2,
    parts_decay: float = 0.5,
    super_optimize: str = "auto-hq",
    seed: int | None = None,
    **partition_opts: Any,
) -> tuple[CompressedIR, ctg.ContractionTree]:
    # ----------------------------
    # Prepare circuit
    # ----------------------------
    circuit = remove_operations_by_name(circuit, {"barrier"}, inplace=False)
    hir = HybridCircuitIR(circuit)

    match compress:
        case "2q":
            ir = compress_2q_gates(hir)
        case "qubits":
            ir = compress_qubits(hir)
        case "none":
            ir = CompressedIR(hir, set())
        case _:
            raise ValueError(f"Unknown compression {compress}")

    choose_leaf_fn = max_nodes_leaf if choose_leaf == "nodes" else max_qubits_leaf
    tree = ir.contraction_tree()

    # ----------------------------
    # Setup randomization
    # ----------------------------
    rng = ctg.core.get_rng(seed)
    rand_size_dict = ctg.core.jitter_dict(tree.size_dict.copy(), random_strength, rng)

    # dynamic imbalance logic
    dyn_imb = ("imbalance" in partition_opts) and ("imbalance_decay" in partition_opts)
    if dyn_imb:
        imbalance = partition_opts.pop("imbalance")
        imbalance_decay = partition_opts.pop("imbalance_decay")
    else:
        imbalance = imbalance_decay = None

    dyn_fix = partition_opts.get("fix_output_nodes") == "auto"

    # ----------------------------
    # Main loop
    # ----------------------------
    while True:
        q_old = quantum_cost(ir, tree)
        c_old = max(tree.contraction_cost(), 1)

        if tree.is_complete():
            break
        if c_old >= max_overhead:
            break

        leaf = choose_leaf_fn(ir, tree)
        if leaf is None:
            break

        tree_before = tree.copy()

        partition_and_contract_subgraph(
            tree,
            leaf,
            rand_size_dict,
            rng,
            parts,
            parts_decay,
            dyn_imb,
            imbalance,
            imbalance_decay,
            dyn_fix,
            partition_opts,
            super_optimize,
        )

        q_new = quantum_cost(ir, tree)
        c_new = tree.contraction_cost()

        quantum_ok = q_new <= q_old / gamma_q
        classical_ok = c_new <= c_old * gamma_c

        # print(
        #     f"[Iter] q: {q_old:.1f} → {q_new:.1f} | "
        #     f"c: {c_old:.2e} → {c_new:.2e} | "
        #     f"q_ok={quantum_ok}, c_ok={classical_ok}"
        # )

        if not (quantum_ok and classical_ok):
            tree = tree_before
            break

    return ir, tree


# =============================================================================
#  OPTUNA SELECTION: maximize quantum improvement, minimize classical cost
# =============================================================================
def get_best_trial_by_q_improvement(study: optuna.Study) -> optuna.Trial:
    trials = [t for t in study.trials if t.values is not None]
    if not trials:
        raise ValueError("No completed trials.")

    qs = np.array([t.values[0] for t in trials], float)
    cs = np.array([t.values[1] for t in trials], float)

    baseline_q = float(np.max(qs))
    q_improve = baseline_q - qs

    ranking = sorted(
        range(len(trials)),
        key=lambda i: (
            -q_improve[i],  # maximize improvement
            cs[i],  # minimize classical cost
            qs[i],  # tie-breaker: smallest quantum cost
        ),
    )
    return trials[ranking[0]]


# =============================================================================
#  Optuna Objective
# =============================================================================
def objective(
    trial: optuna.Trial,
    circuit: QuantumCircuit,
    choose_leaf_methods: list[str],
    compression_methods: list[str],
    gamma_q: float = 1.10,
    gamma_c: float = 1000.0,
    max_overhead: float = np.inf,
) -> tuple[float, float]:

    # hyperparameters to explore
    compress = trial.suggest_categorical("compress", compression_methods)
    choose_leaf = trial.suggest_categorical("choose_leaf", choose_leaf_methods)

    random_strength = trial.suggest_float("random_strength", 0.01, 10.0)
    imbalance = trial.suggest_float("imbalance", 0.01, 1.0)
    imbalance_decay = trial.suggest_float("imbalance_decay", -5, 5)

    # Use fixed small partitioning (your original code had this)
    parts = trial.suggest_int("parts", 2, 2)
    parts_decay = trial.suggest_float("parts_decay", 0.0, 1.0)

    gamma_q = trial.suggest_float("gamma_q", 1.01, 2.0, log=True)
    gamma_c = trial.suggest_float("gamma_c", 1.0, 2000.0, log=True)

    ir, tree = optimize(
        circuit,
        gamma_q=gamma_q,
        gamma_c=gamma_c,
        max_overhead=max_overhead,
        choose_leaf=choose_leaf,
        compress=compress,
        random_strength=random_strength,
        weight_edges=trial.suggest_categorical("weight_edges", ["const", "log"]),
        imbalance=imbalance,
        imbalance_decay=imbalance_decay,
        parts=parts,
        parts_decay=parts_decay,
        mode=trial.suggest_categorical("mode", ["direct", "recursive"]),
        objective=trial.suggest_categorical("objective", ["cut", "km1"]),
        fix_output_nodes=trial.suggest_categorical("fix_output_nodes", ["auto", ""]),
    )

    # trial.set_user_attr("ir", ir)
    # trial.set_user_attr("tree", tree)
    _TRIAL_RESULTS[trial.number] = (ir, tree)
    return quantum_cost(ir, tree), tree.contraction_cost()


# =============================================================================
#  Hyper-Optimization Wrapper
# =============================================================================
def hyper_optimize(
    circuit: QuantumCircuit,
    *,
    gamma_q: float = 1.10,
    gamma_c: float = 1000.0,
    num_threads: int = 1,
    max_overhead: float = np.inf,
    n_trials: int = 100,
    show_progress_bar: bool = False,
) -> QuantumCircuit:

    compression_methods = ["none"]
    choose_leaf_methods = ["nodes"]

    try:
        optuna.delete_study(study_name="cut_opt", storage="sqlite:///study.db")
    except KeyError:
        pass

    study = optuna.create_study(
        storage="sqlite:///study.db",
        study_name="cut_opt",
        load_if_exists=True,
        directions=["minimize", "minimize"],
    )

    def func(trial):
        return objective(
            trial,
            circuit=circuit,
            gamma_c=gamma_c,
            gamma_q=gamma_q,
            max_overhead=max_overhead,
            choose_leaf_methods=choose_leaf_methods,
            compression_methods=compression_methods,
        )

    study.optimize(
        func,
        n_trials=n_trials,
        n_jobs=num_threads,
        show_progress_bar=show_progress_bar,
    )
    # study.optimize(
    #     lambda trial: objective(
    #         trial,
    #         circuit=circuit,
    #         max_overhead=max_overhead,
    #         choose_leaf_methods=choose_leaf_methods,
    #         compression_methods=compression_methods,
    #     ),
    #     n_trials=n_trials,
    #     show_progress_bar=show_progress_bar,
    # )

    best_trial = get_best_trial_by_q_improvement(study)
    best_ir, best_tree = _TRIAL_RESULTS[best_trial.number]
    # best_ir = best_trial.user_attrs["ir"]
    # best_tree = best_trial.user_attrs["tree"]

    return best_ir.cut_circuit(get_leafs(best_tree))
