from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, Any, cast

import cotengra as ctg
import numpy as np

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
#  Parallel Trial Execution (ProcessPoolExecutor)
# =============================================================================
def _sample_params(rng: np.random.Generator) -> dict[str, Any]:
    """Sample random hyperparameters for a trial."""
    return {
        "choose_leaf": rng.choice(["nodes"]),
        "random_strength": float(rng.uniform(0.01, 0.2)),
        "imbalance": float(rng.uniform(0.01, 0.3)),
        "imbalance_decay": float(rng.uniform(0, 1)),
        "parts": int(rng.integers(2, 4)),  # 2 or 3
        "parts_decay": float(rng.uniform(0.0, 1.0)),
        "weight_edges": "log",
        "mode": rng.choice(["direct", "recursive"]),
        "objective": rng.choice(["cut", "km1"]),
        "fix_output_nodes": rng.choice(["auto", ""]),
    }


def _run_trial_in_process(args: tuple) -> dict[str, Any]:
    """Run a single optimization trial in a separate process.

    This function is designed to be called via ProcessPoolExecutor to bypass
    the GIL limitation of KaHyPar (which is CPU-bound and doesn't release GIL).
    
    Returns intermediate results at each partition step to build Pareto frontier.
    """
    circuit, max_sampling_cost, params, trial_id = args

    # Import inside the process to avoid pickling issues
    from qtpu.compiler._ir import HybridCircuitIR
    from qtpu.compiler._util import get_leafs, sampling_overhead_tree
    from qtpu.transforms import remove_operations_by_name
    
    import cotengra as ctg
    import numpy as np

    circuit = remove_operations_by_name(circuit, {"barrier"}, inplace=False)
    ir = HybridCircuitIR(circuit)
    
    choose_leaf_fn = max_nodes_leaf if params["choose_leaf"] == "nodes" else max_qubits_leaf
    tree = ir.contraction_tree()

    # Setup randomization
    rng = ctg.core.get_rng(None)
    rand_size_dict = ctg.core.jitter_dict(tree.size_dict.copy(), params["random_strength"], rng)

    # dynamic imbalance logic
    imbalance = params["imbalance"]
    imbalance_decay = params["imbalance_decay"]
    dyn_fix = params.get("fix_output_nodes") == "auto"
    
    partition_opts = {
        "weight_edges": params["weight_edges"],
        "mode": params["mode"],
        "objective": params["objective"],
        "fix_output_nodes": params["fix_output_nodes"],
    }

    # Collect snapshots at each step for Pareto frontier
    snapshots = []
    
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
            params["parts"],
            params["parts_decay"],
            True,  # dynamic_imbalance
            imbalance,
            imbalance_decay,
            dyn_fix,
            partition_opts,
            "auto-hq",
        )

        # Record snapshot after each partition
        leafs = get_leafs(tree)
        sampling_cost = sampling_overhead_tree(tree)
        quantum_cost = get_max_subcircuit_width_fast(ir, leafs)
        
        # Compute c_cost (contraction cost) and max_error from actual tensor network
        cut_circuit = ir.cut_circuit(leafs)
        import qtpu
        htn = qtpu.circuit_to_hybrid_tn(cut_circuit)
        c_cost = htn.to_dummy_tn().contraction_cost(optimize="auto")
        
        # Compute max_error: sum of gate errors per subcircuit
        max_error = max(
            sum(0.01 if inst.operation.num_qubits == 2 else 0.001 
                for inst in sc.data)
            for sc in htn.subcircuits
        )
        
        snapshots.append({
            "sampling_cost": sampling_cost,
            "quantum_cost": quantum_cost,
            "c_cost": c_cost,
            "max_error": max_error,
        })
        
        # Stop if we've exceeded max_sampling_cost
        if max_sampling_cost is not None and sampling_cost > max_sampling_cost:
            break

    return {
        "trial_id": trial_id,
        "params": params,
        "snapshots": snapshots,
    }


# =============================================================================
#  Hyper-Optimization Wrapper
# =============================================================================
def _compute_pareto_frontier(points: list[dict], metric: str = "quantum_cost") -> list[dict]:
    """Compute the Pareto frontier from a set of points.
    
    A point is Pareto-optimal if no other point has both lower c_cost and lower quantum_cost.
    We want to minimize both quantum_cost and c_cost.
    """
    if not points:
        return []
    
    # Sort by c_cost, then by quantum_cost
    sorted_points = sorted(points, key=lambda x: (x["c_cost"], x["quantum_cost"]))
    
    frontier = []
    best_quantum_cost = float('inf')
    
    for point in sorted_points:
        if point["quantum_cost"] < best_quantum_cost:
            frontier.append(point)
            best_quantum_cost = point["quantum_cost"]
    
    return frontier


def hyper_optimize(
    circuit: QuantumCircuit,
    *,
    max_sampling_cost: float | None = None,
    num_workers: int | None = None,
    n_trials: int = 100,
    seed: int | None = None,
    show_progress_bar: bool = False,
) -> QuantumCircuit:
    """Optimize circuit cutting to minimize quantum cost under a sampling cost constraint.

    Uses ProcessPoolExecutor for true parallelism since KaHyPar (the partitioner)
    is CPU-bound and doesn't release the Python GIL. This provides significant
    speedup on multi-core machines.

    Args:
        circuit: The quantum circuit to optimize.
        max_sampling_cost: Maximum allowed sampling cost. If None, explores the
            full Pareto frontier and returns the solution with best quantum cost.
        num_workers: Number of parallel worker processes. Defaults to min(8, cpu_count).
        n_trials: Number of optimization trials to run.
        seed: Random seed for reproducibility.
        show_progress_bar: Whether to show optimization progress (not yet implemented for parallel).

    Returns:
        The cut circuit with minimal quantum cost under the sampling constraint.
    """
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 1)

    # Sample hyperparameters for all trials upfront
    rng = np.random.default_rng(seed)
    all_params = [_sample_params(rng) for _ in range(n_trials)]

    # Run trials in parallel using ProcessPoolExecutor
    # This bypasses GIL since each trial runs in a separate process
    args_list = [
        (circuit, max_sampling_cost, params, i) for i, params in enumerate(all_params)
    ]

    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_run_trial_in_process, args_list))
    else:
        # Sequential fallback
        results = [_run_trial_in_process(args) for args in args_list]

    # Collect all snapshots from all trials
    all_points = []
    for result in results:
        for snapshot in result["snapshots"]:
            all_points.append({
                "sampling_cost": snapshot["sampling_cost"],
                "quantum_cost": snapshot["quantum_cost"],
                "trial_id": result["trial_id"],
                "params": result["params"],
            })
    
    # Compute Pareto frontier
    frontier = _compute_pareto_frontier(all_points)
    
    if not frontier:
        raise ValueError("No valid solutions found.")
    
    # Select best point based on constraint
    if max_sampling_cost is not None:
        valid_points = [p for p in frontier if p["sampling_cost"] <= max_sampling_cost]
        if not valid_points:
            raise ValueError(f"No solutions found with sampling_cost <= {max_sampling_cost}")
        best = min(valid_points, key=lambda x: x["quantum_cost"])
    else:
        # No constraint - return solution with best quantum cost
        best = min(frontier, key=lambda x: x["quantum_cost"])
    
    logger.info(
        f"Best solution: quantum_cost={best['quantum_cost']}, "
        f"sampling_cost={best['sampling_cost']}"
    )

    # Re-run the best trial to get the actual cut circuit
    circuit = remove_operations_by_name(circuit, {"barrier"}, inplace=False)
    ir = HybridCircuitIR(circuit)
    ir, tree = optimize(
        ir=ir,
        max_sampling_cost=best["sampling_cost"],
        **best["params"],
    )

    return ir.cut_circuit(get_leafs(tree))


def get_pareto_frontier(
    circuit: QuantumCircuit,
    *,
    max_sampling_cost: float = 100,
    num_workers: int | None = None,
    n_trials: int = 100,
    seed: int | None = None,
) -> list[dict]:
    """Find the Pareto frontier of quantum cost vs sampling cost tradeoffs.
    
    This explores the space of possible cuts up to max_sampling_cost and returns 
    all Pareto-optimal solutions, allowing users to choose their preferred tradeoff.
    
    Args:
        circuit: The quantum circuit to optimize.
        max_sampling_cost: Maximum sampling cost to explore. Defaults to 200.
        num_workers: Number of parallel worker processes. Defaults to min(8, cpu_count).
        n_trials: Number of optimization trials to run.
        seed: Random seed for reproducibility.
        
    Returns:
        List of Pareto-optimal points, each with 'quantum_cost' and 'sampling_cost'.
    """
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 1)

    rng = np.random.default_rng(seed)
    all_params = [_sample_params(rng) for _ in range(n_trials)]

    args_list = [
        (circuit, max_sampling_cost, params, i) for i, params in enumerate(all_params)
    ]

    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_run_trial_in_process, args_list))
    else:
        results = [_run_trial_in_process(args) for args in args_list]

    # Add the "no cut" baseline - full circuit with no classical overhead
    circuit_clean = remove_operations_by_name(circuit, {"barrier"}, inplace=False)
    no_cut_error = sum(
        0.01 if inst.operation.num_qubits == 2 else 0.001 
        for inst in circuit_clean.data
    )
    all_points = [{
        "quantum_cost": circuit_clean.num_qubits,
        "c_cost": 0,
        "max_error": no_cut_error,
        "sampling_cost": 0,
    }]
    
    # Collect all snapshots from optimization trials
    for result in results:
        for snapshot in result["snapshots"]:
            all_points.append({
                "quantum_cost": snapshot["quantum_cost"],
                "c_cost": snapshot["c_cost"],
                "max_error": snapshot["max_error"],
                "sampling_cost": snapshot["sampling_cost"],
            })
    
    return _compute_pareto_frontier(all_points, metric="quantum_cost")


def cut_at_target(
    circuit: QuantumCircuit,
    target_quantum_cost: int,
    num_workers: int | None = None,
    n_trials: int = 100,
    seed: int | None = None,
) -> QuantumCircuit:
    """Cut circuit to achieve a specific target quantum cost (max subcircuit width).
    
    Args:
        circuit: The quantum circuit to cut.
        target_quantum_cost: Target max subcircuit width to achieve.
        num_workers: Number of parallel workers.
        n_trials: Number of optimization trials.
        seed: Random seed.
        
    Returns:
        Cut circuit with quantum_cost <= target_quantum_cost.
    """
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 1)

    rng = np.random.default_rng(seed)
    all_params = [_sample_params(rng) for _ in range(n_trials)]

    args_list = [
        (circuit, 200, params, i) for i, params in enumerate(all_params)
    ]

    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_run_trial_in_process, args_list))
    else:
        results = [_run_trial_in_process(args) for args in args_list]

    # Find the snapshot that achieves target_quantum_cost with lowest c_cost
    best_snapshot = None
    best_c_cost = float('inf')
    best_params = None
    
    for result in results:
        for snapshot in result["snapshots"]:
            if snapshot["quantum_cost"] <= target_quantum_cost:
                if snapshot["c_cost"] < best_c_cost:
                    best_c_cost = snapshot["c_cost"]
                    best_snapshot = snapshot
                    best_params = result["params"]
    
    if best_snapshot is None:
        raise ValueError(f"Could not achieve quantum_cost <= {target_quantum_cost}")
    
    # Re-run with best params to get the circuit
    circuit = remove_operations_by_name(circuit, {"barrier"}, inplace=False)
    ir = HybridCircuitIR(circuit)
    
    # Find the sampling_cost that gives us this quantum_cost
    ir, tree = optimize(
        ir=ir,
        max_sampling_cost=best_snapshot["sampling_cost"],
        **best_params,
    )

    return ir.cut_circuit(get_leafs(tree))
