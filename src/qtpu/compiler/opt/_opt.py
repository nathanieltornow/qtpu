from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import cotengra as ctg
import numpy as np

from qtpu.compiler.opt._ir import HybridCircuitIR
from qtpu.compiler.opt._util import get_leafs, sampling_overhead_tree
from qtpu.transforms import remove_operations_by_name

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


@dataclass
class CutPoint:
    """A single point in the optimization space."""

    c_cost: float  # Classical contraction cost (FLOPs)
    max_error: float  # Estimated error in largest subcircuit
    max_size: int  # Max subcircuit width (qubits)
    sampling_cost: float  # Sum of cut edge weights (QPD overhead)
    leafs: list[list[int]] | None  # Partition (None = no cut)


@dataclass
class OptimizationResult:
    """Result of circuit cutting optimization."""

    # All collected snapshots with metrics
    all_points: list[CutPoint]

    # Pareto-optimal points (c_cost vs max_error)
    pareto_frontier: list[CutPoint]

    # The IR for reconstructing circuits
    ir: HybridCircuitIR

    def filter(
        self,
        max_size: int | None = None,
        max_c_cost: float | None = None,
        max_error: float | None = None,
    ) -> list[CutPoint]:
        """Filter Pareto frontier by constraints."""
        valid = self.pareto_frontier.copy()

        if max_size is not None:
            valid = [p for p in valid if p.max_size <= max_size]
        if max_c_cost is not None:
            valid = [p for p in valid if p.c_cost <= max_c_cost]
        if max_error is not None:
            valid = [p for p in valid if p.max_error <= max_error]

        return valid

    def select_best(
        self,
        cost_weight: float = 1.0,
        max_size: int | None = None,
        max_c_cost: float | None = None,
    ) -> CutPoint | None:
        """Select best point using utility function.

        Minimizes: normalized_max_error + cost_weight * normalized_c_cost
        """
        valid = self.filter(max_size=max_size, max_c_cost=max_c_cost)

        if not valid:
            return None
        if len(valid) == 1:
            return valid[0]

        min_cost = min(p.c_cost for p in valid)
        max_cost = max(p.c_cost for p in valid)
        min_error = min(p.max_error for p in valid)
        max_error = max(p.max_error for p in valid)

        cost_range = max_cost - min_cost if max_cost > min_cost else 1
        error_range = max_error - min_error if max_error > min_error else 1

        def score(p):
            norm_cost = (p.c_cost - min_cost) / cost_range
            norm_error = (p.max_error - min_error) / error_range
            return norm_error + cost_weight * norm_cost

        return min(valid, key=score)

    def get_cut_circuit(self, point: CutPoint) -> QuantumCircuit:
        """Get the cut circuit for a specific point.

        Args:
            point: A CutPoint from the frontier.

        Returns:
            The cut quantum circuit.
        """
        if point.leafs is None:
            # No-cut case
            return self.ir.circuit.copy()

        leafs = [frozenset(leaf) for leaf in point.leafs]
        return self.ir.cut_circuit(leafs)

    def get_all_cut_circuits(
        self,
        max_size: int | None = None,
        max_c_cost: float | None = None,
        max_error: float | None = None,
    ) -> list[tuple[CutPoint, QuantumCircuit]]:
        """Get all cut circuits from the Pareto frontier.

        Args:
            max_size: Filter by max subcircuit width.
            max_c_cost: Filter by max classical cost.
            max_error: Filter by max error.

        Returns:
            List of (CutPoint, QuantumCircuit) tuples.
        """
        valid = self.filter(
            max_size=max_size,
            max_c_cost=max_c_cost,
            max_error=max_error,
        )
        return [(p, self.get_cut_circuit(p)) for p in valid]


def get_max_error_from_leafs(ir: HybridCircuitIR, leafs: list[frozenset[int]]) -> float:
    """Compute max subcircuit error directly from IR and leafs.
    
    This avoids the expensive circuit_to_heinsum call by computing error
    directly from node info.
    """
    max_error = 0.0
    # Access internal circuit directly to avoid copy on each access
    circuit_data = ir._circuit.data
    
    for leaf in leafs:
        # Get unique operations in this subcircuit
        op_indices = set()
        for node in leaf:
            info = ir.node_info(node)
            op_indices.add(info.op_idx)
        
        # Sum errors for operations in this subcircuit
        subcircuit_error = 0.0
        for op_idx in op_indices:
            instr = circuit_data[op_idx]
            if instr.operation.num_qubits == 2:
                subcircuit_error += 0.01
            elif instr.operation.num_qubits == 1:
                subcircuit_error += 0.001
        
        max_error = max(max_error, subcircuit_error)
    
    return max_error


def get_max_subcircuit_width_fast(
    ir: HybridCircuitIR, leafs: list[frozenset[int]]
) -> int:
    """Fast computation of max subcircuit width directly from IR structure."""
    max_width = 0

    hg = ir._hypergraph
    node_neighbors: dict[int, set[int]] = {i: set() for i in range(hg.num_nodes)}
    for edge_nodes in hg.edges.values():
        if len(edge_nodes) == 2:
            u, v = edge_nodes
            node_neighbors[u].add(v)
            node_neighbors[v].add(u)

    for leaf in leafs:
        nodes = set(leaf)
        qubit_nodes: dict = {}
        for node in nodes:
            info = ir.node_info(node)
            if info.abs_qubit not in qubit_nodes:
                qubit_nodes[info.abs_qubit] = []
            qubit_nodes[info.abs_qubit].append((node, info.op_idx))

        width = 0
        for qubit, nodes_with_time in qubit_nodes.items():
            nodes_with_time.sort(key=lambda x: x[1])
            segments = 1
            for i in range(len(nodes_with_time) - 1):
                node1 = nodes_with_time[i][0]
                node2 = nodes_with_time[i + 1][0]
                if node2 not in node_neighbors[node1]:
                    segments += 1
            width += segments

        max_width = max(max_width, width)

    return max_width


def get_max_subcircuit_nodes(leafs: list[frozenset[int]]) -> int:
    """Get the maximum number of nodes in any subcircuit."""
    return max(len(leaf) for leaf in leafs) if leafs else 0


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
        "imbalance": float(rng.uniform(0.01, 0.6)),  # Keep <= 0.5 to avoid deep trees
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
    from qtpu.compiler.opt._ir import HybridCircuitIR
    from qtpu.compiler.opt._util import get_leafs, sampling_overhead_tree
    from qtpu.transforms import remove_operations_by_name

    import cotengra as ctg

    circuit = remove_operations_by_name(circuit, {"barrier"}, inplace=False)
    ir = HybridCircuitIR(circuit)

    tree = ir.contraction_tree()

    # Setup randomization — seed with trial_id so repeated runs of
    # get_pareto_frontier with the same outer `seed` (which fixes the
    # hyperparameter sequence via _sample_params) produce identical
    # frontiers trial-for-trial.
    rng = ctg.core.get_rng(trial_id)
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
        size = get_max_subcircuit_width_fast(ir, leafs)

        # Track contraction cost directly from tree
        # This is the cost of contracting the partitioning tree
        try:
            c_cost = tree.contraction_cost()
        except Exception:
            c_cost = sampling_cost  # Fallback

        snapshots.append(
            {
                "sampling_cost": sampling_cost,
                "size": size,
                "c_cost": c_cost,
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
def _compute_pareto_frontier(points: list[CutPoint]) -> list[CutPoint]:
    """Compute the Pareto frontier on c_cost vs max_error.

    A point is Pareto-optimal if no other point has both lower c_cost
    and lower max_error. We want to minimize both.
    """
    if not points:
        return []

    # Sort by c_cost (ascending), then by max_error (ascending)
    sorted_points = sorted(points, key=lambda x: (x.c_cost, x.max_error))

    frontier = []
    best_max_error = float("inf")

    for point in sorted_points:
        if point.max_error < best_max_error:
            frontier.append(point)
            best_max_error = point.max_error

    return frontier


# =============================================================================
#  Main API
# =============================================================================
def get_pareto_frontier(
    circuit: QuantumCircuit,
    *,
    max_sampling_cost: float = 120,
    num_workers: int | None = None,
    n_trials: int = 100,
    seed: int | None = None,
) -> OptimizationResult:
    """Find the Pareto frontier of c_cost vs max_error tradeoffs.

    Explores the space of possible cuts and returns Pareto-optimal solutions
    where the frontier is computed on c_cost (contraction cost) vs max_error
    (estimated error in largest subcircuit).

    Args:
        circuit: The quantum circuit to optimize.
        max_sampling_cost: Maximum sampling cost to explore.
        num_workers: Number of parallel workers. Defaults to min(8, cpu_count).
        n_trials: Number of optimization trials.
        seed: Random seed for reproducibility.

    Returns:
        OptimizationResult with:
        - all_points: All collected optimization snapshots
        - pareto_frontier: Pareto-optimal points on c_cost vs max_error
        - ir: HybridCircuitIR for reconstruction
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

    # Setup IR for error computation
    circuit_clean = remove_operations_by_name(circuit, {"barrier"}, inplace=False)
    ir = HybridCircuitIR(circuit_clean)

    # Compute no-cut baseline error
    no_cut_error = sum(
        0.01 if inst.operation.num_qubits == 2 else 0.001 for inst in circuit_clean.data
    )

    # Add the "no cut" baseline
    all_points: list[CutPoint] = [
        CutPoint(
            c_cost=0,  # No cuts = no classical overhead
            max_error=no_cut_error,
            max_size=circuit_clean.num_qubits,
            sampling_cost=0,
            leafs=None,  # Special marker for no-cut baseline
        )
    ]

    # Collect all snapshots and compute max_error for each (fast path)
    for result in results:
        for snapshot in result["snapshots"]:
            leafs = [frozenset(leaf) for leaf in snapshot["leafs"]]

            # Compute max_error directly from IR (fast - no circuit reconstruction)
            max_error = get_max_error_from_leafs(ir, leafs)

            all_points.append(
                CutPoint(
                    c_cost=snapshot.get("c_cost", snapshot["sampling_cost"]),
                    max_error=max_error,
                    max_size=snapshot["size"],
                    sampling_cost=snapshot["sampling_cost"],
                    leafs=snapshot["leafs"],
                )
            )

    # Compute Pareto frontier on c_cost vs max_error
    frontier = _compute_pareto_frontier(all_points)

    return OptimizationResult(
        all_points=all_points,
        pareto_frontier=frontier,
        ir=ir,
    )
