from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, Any, cast

import cotengra as ctg
import numpy as np

from qtpu.compiler._ir import HybridCircuitIR
from qtpu.compiler._util import get_leafs, sampling_overhead_tree
from qtpu.transforms import remove_operations_by_name

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


# =============================================================================
#  Fast metrics computation
# =============================================================================
def get_max_subcircuit_width_fast(
    ir: HybridCircuitIR, leafs: list[frozenset[int]]
) -> int:
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


# =============================================================================
#  Leaf selection
# =============================================================================
def max_qubits_leaf(ir: HybridCircuitIR, tree: ctg.ContractionTree):
    """Select the leaf with the most qubits for partitioning."""
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
#  Parallel Trial Execution (ProcessPoolExecutor)
# =============================================================================
def _sample_params(rng: np.random.Generator) -> dict[str, Any]:
    """Sample random hyperparameters for a trial."""
    return {
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

    tree = ir.contraction_tree()

    # Setup randomization
    rng = ctg.core.get_rng(None)
    rand_size_dict = ctg.core.jitter_dict(
        tree.size_dict.copy(), params["random_strength"], rng
    )

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

        leaf = max_qubits_leaf(ir, tree)
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

        # Record snapshot after each partition (fast metrics only)
        leafs = get_leafs(tree)
        sampling_cost = sampling_overhead_tree(tree)
        quantum_cost = get_max_subcircuit_width_fast(ir, leafs)

        snapshots.append(
            {
                "sampling_cost": sampling_cost,
                "quantum_cost": quantum_cost,
                "leafs": [list(leaf) for leaf in leafs],  # Store partition for later
            }
        )

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
def _compute_pareto_frontier(
    points: list[dict], metric: str = "quantum_cost"
) -> list[dict]:
    """Compute the Pareto frontier from a set of points.

    A point is Pareto-optimal if no other point has both higher sampling_cost
    and higher quantum_cost. We want to minimize both.

    Uses sampling_cost as the "cost" axis since it's available during optimization.
    """
    if not points:
        return []

    # Sort by sampling_cost, then by quantum_cost
    sorted_points = sorted(
        points, key=lambda x: (x["sampling_cost"], x["quantum_cost"])
    )

    frontier = []
    best_quantum_cost = float("inf")

    for point in sorted_points:
        if point["quantum_cost"] < best_quantum_cost:
            frontier.append(point)
            best_quantum_cost = point["quantum_cost"]

    return frontier


# =============================================================================
#  Main API
# =============================================================================
def get_pareto_frontier(
    circuit: QuantumCircuit,
    *,
    max_sampling_cost: float = 200,
    num_workers: int | None = None,
    n_trials: int = 100,
    seed: int | None = None,
) -> list[dict]:
    """Find the Pareto frontier of quantum cost vs classical cost tradeoffs.

    Explores the space of possible cuts and returns all Pareto-optimal solutions.
    Each point includes quantum_cost (max subcircuit width), c_cost (contraction cost),
    max_error (estimated error), and leafs (partition for reconstruction).

    Args:
        circuit: The quantum circuit to optimize.
        max_sampling_cost: Maximum sampling cost to explore.
        num_workers: Number of parallel workers. Defaults to min(8, cpu_count).
        n_trials: Number of optimization trials.
        seed: Random seed for reproducibility.

    Returns:
        List of Pareto-optimal points with keys:
        - quantum_cost: Max subcircuit width (qubits)
        - c_cost: Classical contraction cost
        - max_error: Estimated max subcircuit error
        - leafs: Partition (for circuit reconstruction)
    """
    import qtpu

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
    ir = HybridCircuitIR(circuit_clean)
    no_cut_error = sum(
        0.01 if inst.operation.num_qubits == 2 else 0.001 for inst in circuit_clean.data
    )

    all_points = [
        {
            "quantum_cost": circuit_clean.num_qubits,
            "sampling_cost": 0,
            "leafs": None,  # Special marker for no-cut baseline
        }
    ]

    # Collect all snapshots from optimization trials (fast metrics only)
    for result in results:
        for snapshot in result["snapshots"]:
            all_points.append(
                {
                    "quantum_cost": snapshot["quantum_cost"],
                    "sampling_cost": snapshot["sampling_cost"],
                    "leafs": snapshot["leafs"],
                }
            )

    # Compute Pareto frontier based on fast metrics (sampling_cost, quantum_cost)
    frontier_points = _compute_pareto_frontier(all_points, metric="quantum_cost")

    # Now compute expensive metrics (c_cost, max_error) only for Pareto-optimal points
    frontier_with_metrics = []
    for point in frontier_points:
        if point["leafs"] is None:
            # No-cut baseline
            frontier_with_metrics.append(
                {
                    "quantum_cost": point["quantum_cost"],
                    "sampling_cost": point["sampling_cost"],
                    "c_cost": 0,
                    "max_error": no_cut_error,
                    "leafs": None,  # Include leafs for later use
                }
            )
        else:
            # Compute c_cost and max_error from the stored partition
            leafs = [frozenset(leaf) for leaf in point["leafs"]]
            cut_circuit = ir.cut_circuit(leafs)
            htn = qtpu.circuit_to_hybrid_tn(cut_circuit)
            c_cost = htn.to_dummy_tn().contraction_cost(optimize="auto")
            max_error = max(
                sum(
                    0.01 if inst.operation.num_qubits == 2 else 0.001
                    for inst in sc.data
                )
                for sc in htn.subcircuits
            )
            frontier_with_metrics.append(
                {
                    "quantum_cost": point["quantum_cost"],
                    "sampling_cost": point["sampling_cost"],
                    "c_cost": c_cost,
                    "max_error": max_error,
                    "leafs": point["leafs"],  # Include leafs for later use
                }
            )

    return frontier_with_metrics
