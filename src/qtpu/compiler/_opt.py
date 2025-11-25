from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import cotengra as ctg
import numpy as np
import optuna

from qtpu.compiler._ir import HybridCircuitIR
from qtpu.compiler._util import get_leafs
from qtpu.transforms import remove_operations_by_name

from ._util import sampling_overhead_tree


def get_max_subcircuit_width_fast(ir: HybridCircuitIR, leafs: list[frozenset[int]]) -> int:
    """Fast computation of max subcircuit width directly from IR structure.
    
    For each leaf partition:
    - Count unique physical qubits
    - Add 1 for each wire cut (same qubit appearing in different time slices
      within this partition due to cuts)
    
    This avoids generating the full cut circuit.
    """
    max_width = 0
    
    # Build edge lookup: node -> set of connected nodes
    hg = ir._hypergraph
    node_neighbors: dict[int, set[int]] = {i: set() for i in range(hg.num_nodes)}
    for edge_nodes in hg.edges.values():
        if len(edge_nodes) == 2:
            u, v = edge_nodes
            node_neighbors[u].add(v)
            node_neighbors[v].add(u)
    
    for leaf in leafs:
        nodes = set(leaf)
        
        # Group nodes by qubit
        qubit_nodes: dict = {}
        for node in nodes:
            info = ir.node_info(node)
            if info.abs_qubit not in qubit_nodes:
                qubit_nodes[info.abs_qubit] = []
            qubit_nodes[info.abs_qubit].append((node, info.op_idx))
        
        # For each qubit, count segments (breaks due to wire cuts)
        width = 0
        for qubit, nodes_with_time in qubit_nodes.items():
            # Sort by time
            nodes_with_time.sort(key=lambda x: x[1])
            
            # Count segments - each break due to a cut adds a new "qubit" 
            segments = 1
            for i in range(len(nodes_with_time) - 1):
                node1 = nodes_with_time[i][0]
                node2 = nodes_with_time[i + 1][0]
                
                # Check if there's a direct edge between them
                if node2 not in node_neighbors[node1]:
                    # Wire was cut between these nodes
                    segments += 1
            
            width += segments
        
        max_width = max(max_width, width)
    
    return max_width


if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit

logger = logging.getLogger("qtpu.compiler")

if True:  # silence Optuna unless debugging
    optuna.logging.set_verbosity(optuna.logging.WARNING)


_TRIAL_RESULTS: dict[int, tuple[HybridCircuitIR, ctg.ContractionTree]] = {}


# =============================================================================
#  Leaf selection strategies
# =============================================================================
def max_nodes_leaf(ir: HybridCircuitIR, tree: ctg.ContractionTree):
    if len(tree.childless) == 1:
        return cast(frozenset[int], next(iter(tree.childless)))
    if not tree.childless:
        return None
    return cast(frozenset[int], max(tree.childless, key=len))


def max_qubits_leaf(ir: HybridCircuitIR, tree: ctg.ContractionTree):
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
    circuit: QuantumCircuit | None = None,
    *,
    ir: HybridCircuitIR | None = None,
    max_sampling_cost: float = np.inf,
    choose_leaf: str = "nodes",
    random_strength: float = 0.01,
    parts: int = 2,
    parts_decay: float = 0.5,
    super_optimize: str = "auto-hq",
    seed: int | None = None,
    **partition_opts: Any,
) -> tuple[HybridCircuitIR, ctg.ContractionTree]:
    # ----------------------------
    # Prepare circuit / use cached IR
    # ----------------------------
    if ir is None:
        if circuit is None:
            raise ValueError("Either circuit or ir must be provided")
        circuit = remove_operations_by_name(circuit, {"barrier"}, inplace=False)
        ir = HybridCircuitIR(circuit)

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
    leafs = get_leafs(tree)
    best_quantum_cost = get_max_subcircuit_width_fast(ir, leafs)
    
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
        
        # Track the best quantum cost found within budget (fast computation)
        leafs = get_leafs(tree)
        current_quantum_cost = get_max_subcircuit_width_fast(ir, leafs)
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
    cached_ir: HybridCircuitIR,
    choose_leaf_methods: list[str],
    max_sampling_cost: float = np.inf,
) -> float:

    # hyperparameters to explore
    choose_leaf = trial.suggest_categorical("choose_leaf", choose_leaf_methods)

    random_strength = trial.suggest_float("random_strength", 0.01, 0.2)
    # Low imbalance values (0.01-0.3) produce balanced partitions which is critical
    # for achieving low quantum cost (max subcircuit width)
    imbalance = trial.suggest_float("imbalance", 0.01, 0.3)
    imbalance_decay = trial.suggest_float("imbalance_decay", 0, 1)

    # Use fixed small partitioning
    parts = trial.suggest_int("parts", 2, 3)
    parts_decay = trial.suggest_float("parts_decay", 0.0, 1.0)

    ir, tree = optimize(
        ir=cached_ir,
        max_sampling_cost=max_sampling_cost,
        choose_leaf=choose_leaf,
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
    
    # Fast computation of quantum cost directly from IR
    quantum_cost = get_max_subcircuit_width_fast(ir, leafs)

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
    choose_leaf_methods = ["nodes"]

    # Build IR once - this is the expensive part
    circuit = remove_operations_by_name(circuit, {"barrier"}, inplace=False)
    cached_ir = HybridCircuitIR(circuit)

    # Use in-memory storage to avoid database state issues
    study = optuna.create_study(directions=["minimize"])

    def func(trial):
        return objective(
            trial,
            cached_ir=cached_ir,
            max_sampling_cost=max_sampling_cost,
            choose_leaf_methods=choose_leaf_methods,
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
