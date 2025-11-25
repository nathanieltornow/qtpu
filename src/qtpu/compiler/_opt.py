from __future__ import annotations

import logging
from itertools import combinations
from typing import TYPE_CHECKING, Any, cast

import cotengra as ctg
import networkx as nx
import numpy as np
import optuna
from qiskit_addon_cutting.qpd import TwoQubitQPDGate

from qtpu.compiler._compress import CompressedIR, compress_2q_gates, compress_qubits
from qtpu.compiler._ir import HybridCircuitIR
from qtpu.compiler._util import get_leafs
from qtpu.transforms import remove_operations_by_name, wire_cuts_to_moves

from ._util import sampling_overhead_tree


def get_max_subcircuit_width(cut_circuit) -> int:
    """Compute max subcircuit width using qubit graph connected components.
    
    This matches the logic in transforms.py: qubits are connected if they
    share a gate that is NOT a TwoQubitQPDGate (i.e., not a cut).
    """
    circuit = wire_cuts_to_moves(cut_circuit)
    
    # Build qubit graph - same logic as _qubit_graph in transforms.py
    graph = nx.Graph()
    graph.add_nodes_from(circuit.qubits)
    for instr in circuit:
        if instr.operation.name == "barrier" or isinstance(
            instr.operation, TwoQubitQPDGate
        ):
            continue
        qubits = instr.qubits
        for qubit1, qubit2 in combinations(qubits, 2):
            graph.add_edge(qubit1, qubit2)
    
    # Find connected components - each is a subcircuit
    ccs = list(nx.connected_components(graph))
    return max(len(cc) for cc in ccs)

if TYPE_CHECKING:
    from collections.abc import Callable
    from qiskit.circuit import QuantumCircuit

logger = logging.getLogger("qtpu.compiler")

if True:  # silence Optuna unless debugging
    optuna.logging.set_verbosity(optuna.logging.WARNING)


_TRIAL_RESULTS = {}


def max_cost_leaf(ir: CompressedIR, tree: ctg.ContractionTree) -> frozenset[int] | None:
    if len(tree.childless) == 1:
        return cast(frozenset[int], next(iter(tree.childless)))
    if not tree.childless:
        return None
    return cast(
        frozenset[int],
        max(
            tree.childless,
            key=lambda x: len(ir.decompress_nodes(x)),
        ),
    )


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
    return new_subgs


# =============================================================================
#  OPTIMIZE (partition until max sampling cost or tree complete)
# =============================================================================
def optimize(
    circuit: QuantumCircuit,
    *,
    max_sampling_cost: float = np.inf,
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
    # Main loop: partition until max_sampling_cost reached
    # Track the best quantum cost found while staying within budget
    # ----------------------------
    best_tree = tree.copy()
    best_quantum_cost = get_max_subcircuit_width(ir.cut_circuit(get_leafs(tree)))
    
    while True:
        if tree.is_complete():
            break

        leaf = choose_leaf_fn(ir, tree)
        if leaf is None:
            break

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

        # Check if we exceeded the max sampling cost
        current_sampling_cost = sampling_overhead_tree(tree)
        if current_sampling_cost > max_sampling_cost:
            # Stop - we've hit the budget limit
            break
        
        # Track the best quantum cost found within budget
        current_quantum_cost = get_max_subcircuit_width(ir.cut_circuit(get_leafs(tree)))
        if current_quantum_cost <= best_quantum_cost:
            best_quantum_cost = current_quantum_cost
            best_tree = tree.copy()

    # Return the best tree found within budget
    return ir, best_tree


# =============================================================================
#  OPTUNA SELECTION: maximize quantum improvement, minimize classical cost
# =============================================================================
def get_best_trial_by_min_quantum_cost(
    study: optuna.Study, max_sampling_cost: float = np.inf
) -> optuna.Trial:
    """Select the trial with minimal quantum cost that satisfies the sampling cost constraint."""
    trials = [t for t in study.trials if t.values is not None]
    if not trials:
        raise ValueError("No completed trials.")

    # Filter trials that satisfy the sampling cost constraint
    valid_trials = [
        t
        for t in trials
        if t.user_attrs.get("sampling_cost", np.inf) <= max_sampling_cost
    ]

    if not valid_trials:
        raise ValueError("No trials satisfy the sampling cost constraint.")
        # If no trial satisfies the constraint, fall back to the one with lowest sampling cost
        logger.warning(
            f"No trial satisfies max_sampling_cost={max_sampling_cost}. "
            "Selecting trial with lowest sampling cost."
        )
        valid_trials = trials
        # Sort by sampling cost (ascending)
        valid_trials.sort(key=lambda t: t.user_attrs.get("sampling_cost", np.inf))
        return valid_trials[0]

    # Among valid trials, select the one with minimal quantum cost
    valid_trials.sort(key=lambda t: t.user_attrs.get("quantum_cost"))
    print("best trial quantum cost:", valid_trials[0].user_attrs.get("quantum_cost"))
    print("best trial sampling cost:", valid_trials[0].user_attrs.get("sampling_cost"))
    return valid_trials[0]


# =============================================================================
#  Optuna Objective
# =============================================================================
def objective(
    trial: optuna.Trial,
    circuit: QuantumCircuit,
    choose_leaf_methods: list[str],
    compression_methods: list[str],
    max_sampling_cost: float = np.inf,
) -> float:

    # hyperparameters to explore
    compress = trial.suggest_categorical("compress", compression_methods)
    choose_leaf = trial.suggest_categorical("choose_leaf", choose_leaf_methods)

    random_strength = trial.suggest_float("random_strength", 0.01, 0.2)
    # Low imbalance values (0.01-0.3) produce balanced partitions which is critical
    # for achieving low quantum cost (max subcircuit width)
    imbalance = trial.suggest_float("imbalance", 0.01, 0.3)
    imbalance_decay = trial.suggest_float("imbalance_decay", 0, 1)

    # Use fixed small partitioning (your original code had this)
    parts = trial.suggest_int("parts", 2, 3)
    parts_decay = trial.suggest_float("parts_decay", 0.0, 1.0)

    ir, tree = optimize(
        circuit,
        max_sampling_cost=max_sampling_cost,
        choose_leaf=choose_leaf,
        compress=compress,
        random_strength=random_strength,
        weight_edges=trial.suggest_categorical("weight_edges", ["log"]),
        imbalance=imbalance,
        imbalance_decay=imbalance_decay,
        parts=parts,
        parts_decay=parts_decay,
        mode=trial.suggest_categorical("mode", ["direct", "recursive"]),
        objective=trial.suggest_categorical("objective", ["cut", "km1"]),
        fix_output_nodes=trial.suggest_categorical("fix_output_nodes", ["auto", ""]),
    )

    _TRIAL_RESULTS[trial.number] = (ir, tree)

    leafs = get_leafs(tree)
    sampling_cost = sampling_overhead_tree(tree)
    
    # Get actual subcircuit width using qubit graph connected components
    cut_circuit = ir.cut_circuit(leafs)
    quantum_cost = get_max_subcircuit_width(cut_circuit)

    # Store metrics for later selection
    trial.set_user_attr("sampling_cost", sampling_cost)
    trial.set_user_attr("quantum_cost", quantum_cost)

    # Penalize if exceeds max sampling cost
    if sampling_cost > max_sampling_cost:
        return 1e32

    # Minimize quantum cost (primary objective)
    return quantum_cost


# =============================================================================
#  Hyper-Optimization Wrapper
# =============================================================================
def hyper_optimize(
    circuit: QuantumCircuit,
    *,
    max_sampling_cost: float = np.inf,
    num_threads: int = 1,
    n_trials: int = 100,
    show_progress_bar: bool = False,
) -> QuantumCircuit:
    """Optimize circuit cutting to minimize quantum cost under a sampling cost constraint.

    Args:
        circuit: The quantum circuit to optimize.
        max_sampling_cost: Maximum allowed sampling cost.
            Optimization will find the solution with minimal quantum cost
            that stays under this sampling cost threshold.
        num_threads: Number of parallel optimization threads.
        n_trials: Number of Optuna trials to run.
        show_progress_bar: Whether to show optimization progress.

    Returns:
        The cut circuit with minimal quantum cost under the sampling constraint.
    """
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
        directions=["minimize"],
    )

    def func(trial):
        return objective(
            trial,
            circuit=circuit,
            max_sampling_cost=max_sampling_cost,
            choose_leaf_methods=choose_leaf_methods,
            compression_methods=compression_methods,
        )

    study.optimize(
        func,
        n_trials=n_trials,
        n_jobs=num_threads,
        show_progress_bar=show_progress_bar,
    )

    best_trial = get_best_trial_by_min_quantum_cost(study, max_sampling_cost)
    best_ir, best_tree = _TRIAL_RESULTS[best_trial.number]

    return best_ir.cut_circuit(get_leafs(best_tree))
