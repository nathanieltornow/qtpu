from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from qtpu.compiler._opt import get_pareto_frontier, cut_at_target

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


def cut(
    circuit: QuantumCircuit,
    max_subcircuit_error: float | None = None,
    max_classical_cost: float | None = None,
    cost_weight: float | None = None,
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
        cost_weight: How much to weight classical cost vs error reduction (λ).
            Both metrics are normalized to [0, 1] using the Pareto frontier.
            - λ=0: Only minimize error (equivalent to strategy="min_error")
            - λ=1: Equal weight to error and classical cost (default for best_tradeoff)
            - λ>1: Prefer lower classical cost over error reduction
            The score is: normalized_error + λ * normalized_cost
        strategy: How to select from the Pareto frontier:
            - "best_tradeoff": Use cost_weight (default λ=1) to balance error vs cost.
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
    else:  # "best_tradeoff" - use utility function
        lam = cost_weight if cost_weight is not None else 1.0
        best = _select_by_utility(frontier, lam)
    
    # Cut at the target quantum_cost
    return cut_at_target(
        circuit,
        target_quantum_cost=best["quantum_cost"],
        num_workers=num_workers,
        n_trials=n_trials,
        seed=seed,
    )


def _select_by_utility(frontier: list[dict], cost_weight: float) -> dict:
    """Select point from Pareto frontier using utility function.
    
    Minimizes: normalized_error + cost_weight * normalized_cost
    
    Both error and c_cost are normalized to [0, 1] using the frontier's
    own min/max values, so cost_weight is interpretable:
    - cost_weight=0: Only care about error
    - cost_weight=1: Equal weight (1% error reduction = 1% cost increase)
    - cost_weight=2: Cost is 2x as important as error
    
    Tiebreaker: when scores are equal, prefer lower quantum_cost (smaller subcircuits).
    
    Args:
        frontier: List of Pareto-optimal points with 'max_error', 'c_cost', 'quantum_cost'.
        cost_weight: How much to weight classical cost (λ).
        
    Returns:
        The point that minimizes the utility function.
    """
    if len(frontier) == 1:
        return frontier[0]
    
    # Get normalization bounds from frontier
    min_error = min(p["max_error"] for p in frontier)
    max_error = max(p["max_error"] for p in frontier)
    min_cost = min(p["c_cost"] for p in frontier)
    max_cost = max(p["c_cost"] for p in frontier)
    
    error_range = max_error - min_error if max_error > min_error else 1
    cost_range = max_cost - min_cost if max_cost > min_cost else 1
    
    def score(p):
        # Normalize both to [0, 1]
        norm_error = (p["max_error"] - min_error) / error_range
        norm_cost = (p["c_cost"] - min_cost) / cost_range
        # Primary score: utility function
        # Tiebreaker: prefer smaller quantum_cost (add tiny fraction)
        return (norm_error + cost_weight * norm_cost, p["quantum_cost"])
    
    return min(frontier, key=score)


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
