from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from qtpu.compiler._opt import get_pareto_frontier, cut_at_target

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


def cut(
    circuit: QuantumCircuit,
    max_subcircuit_error: float | None = None,
    max_classical_cost: float | None = None,
    strategy: Literal["best_tradeoff", "min_error", "min_cost"] = "best_tradeoff",
    num_workers: int | None = None,
    n_trials: int = 100,
    seed: int | None = None,
) -> QuantumCircuit:
    """Cut a quantum circuit to optimize the quantum-classical tradeoff.

    This function automatically finds an optimal cut that balances quantum error
    reduction against classical computation cost.

    Args:
        circuit: The quantum circuit to cut.
        max_subcircuit_error: Maximum allowed subcircuit error. If specified,
            finds the solution with lowest classical cost under this constraint.
        max_classical_cost: Maximum allowed classical cost (c_cost). If specified,
            finds the solution with lowest error under this constraint.
        strategy: How to select from the Pareto frontier:
            - "best_tradeoff": Automatically pick the knee of the Pareto curve
              (best error reduction per classical cost).
            - "min_error": Minimize error regardless of classical cost.
            - "min_cost": Minimize classical cost regardless of error.
        num_workers: Number of parallel worker processes. Defaults to min(8, cpu_count).
        n_trials: Number of optimization trials to run.
        seed: Random seed for reproducibility.

    Returns:
        The cut circuit optimized according to the specified strategy.
    """
    # Get the Pareto frontier
    frontier = get_pareto_frontier(
        circuit,
        max_sampling_cost=200,  # Explore broadly
        num_workers=num_workers,
        n_trials=n_trials,
        seed=seed,
    )
    
    if not frontier:
        raise ValueError("No valid solutions found.")
    
    # Select best point based on strategy/constraints
    if max_classical_cost is not None:
        # Find lowest error within classical cost budget
        valid = [p for p in frontier if p["c_cost"] <= max_classical_cost]
        if not valid:
            raise ValueError(f"No solutions with c_cost <= {max_classical_cost}")
        best = min(valid, key=lambda p: p["max_error"])
    elif max_subcircuit_error is not None:
        # Find lowest classical cost under error constraint
        valid = [p for p in frontier if p["max_error"] <= max_subcircuit_error]
        if not valid:
            raise ValueError(f"No solutions with max_error <= {max_subcircuit_error}")
        best = min(valid, key=lambda p: p["c_cost"])
    elif strategy == "min_error":
        best = min(frontier, key=lambda p: p["max_error"])
    elif strategy == "min_cost":
        best = min(frontier, key=lambda p: p["c_cost"])
    else:  # "best_tradeoff" - find the knee
        best = _find_knee(frontier)
    
    # Cut at the target quantum_cost
    return cut_at_target(
        circuit,
        target_quantum_cost=best["quantum_cost"],
        num_workers=num_workers,
        n_trials=n_trials,
        seed=seed,
    )


def _find_knee(frontier: list[dict]) -> dict:
    """Find the knee of the Pareto frontier - best error reduction per cost.
    
    Uses the "elbow method" - finds the point with maximum distance from the
    line connecting the first and last points of the frontier.
    """
    if len(frontier) <= 2:
        return frontier[0]
    
    # Sort by c_cost
    sorted_frontier = sorted(frontier, key=lambda p: p["c_cost"])
    
    # Get first and last points
    p1 = (sorted_frontier[0]["c_cost"], sorted_frontier[0]["max_error"])
    p2 = (sorted_frontier[-1]["c_cost"], sorted_frontier[-1]["max_error"])
    
    # Calculate distance from line for each point
    best_point = sorted_frontier[0]
    best_distance = 0
    
    for point in sorted_frontier:
        # Distance from point to line p1-p2
        x0, y0 = point["c_cost"], point["max_error"]
        
        # Normalize to make comparison fair
        x_range = p2[0] - p1[0] if p2[0] != p1[0] else 1
        y_range = p1[1] - p2[1] if p1[1] != p2[1] else 1
        
        x0_norm = (x0 - p1[0]) / x_range
        y0_norm = (p1[1] - y0) / y_range
        x1_norm, y1_norm = 0, 0
        x2_norm, y2_norm = 1, 1
        
        # Distance from point to diagonal line
        distance = abs((y2_norm - y1_norm) * x0_norm - (x2_norm - x1_norm) * y0_norm)
        
        if distance > best_distance:
            best_distance = distance
            best_point = point
    
    return best_point
