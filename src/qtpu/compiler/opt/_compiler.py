from __future__ import annotations

from typing import TYPE_CHECKING

from qtpu.compiler.opt._opt import get_pareto_frontier, CutPoint
from qtpu.transforms import remove_operations_by_name

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


def cut(
    circuit: QuantumCircuit,
    max_size: int | None = None,
    max_c_cost: float | None = None,
    cost_weight: float = 1.0,
    num_workers: int | None = None,
    n_trials: int = 100,
    seed: int | None = None,
) -> QuantumCircuit:
    """Cut a quantum circuit to optimize the quantum-classical tradeoff.

    Finds an optimal cut that balances quantum error reduction against classical
    computation cost, subject to optional constraints.

    Args:
        circuit: The quantum circuit to cut.
        max_size: Maximum allowed subcircuit width (qubits). Hard constraint.
            If None, no qubit constraint is applied.
        max_c_cost: Maximum allowed classical cost. Hard constraint.
            If None, no classical cost constraint is applied.
        cost_weight: How much to weight classical cost vs error reduction (λ).
            Both metrics are normalized to [0, 1] using the Pareto frontier.
            - λ=0: Only minimize error (most aggressive cutting)
            - λ=1: Equal weight to error and classical cost (default)
            - λ>1: Prefer lower classical cost over error reduction
            The score is: normalized_error + λ * normalized_cost
        num_workers: Number of parallel worker processes. Defaults to min(8, cpu_count).
        n_trials: Number of optimization trials to run.
        seed: Random seed for reproducibility.

    Returns:
        The cut circuit optimized according to the cost_weight, subject to constraints.

    Examples:
        # Default: balanced tradeoff, no constraints
        cut_circuit = cut(circuit)

        # Must fit on 10-qubit device, balanced tradeoff
        cut_circuit = cut(circuit, max_size=10)

        # Minimize error aggressively, no constraints
        cut_circuit = cut(circuit, cost_weight=0)

        # Prefer low classical cost, must fit on 15 qubits
        cut_circuit = cut(circuit, max_size=15, cost_weight=2.0)
    """
    # Get the Pareto frontier
    result = get_pareto_frontier(
        circuit,
        max_sampling_cost=150,  # Explore broadly
        num_workers=num_workers,
        n_trials=n_trials,
        seed=seed,
    )

    if not result.pareto_frontier:
        raise ValueError("No valid solutions found.")

    # Apply hard constraints
    valid = result.pareto_frontier

    if max_size is not None:
        valid = [p for p in valid if p.max_size <= max_size]
        if not valid:
            min_achievable = min(p.max_size for p in result.pareto_frontier)
            raise ValueError(
                f"No solutions with max_size <= {max_size}. "
                f"Minimum achievable: {min_achievable} qubits."
            )

    if max_c_cost is not None:
        valid = [p for p in valid if p.c_cost <= max_c_cost]
        if not valid:
            raise ValueError(f"No solutions with c_cost <= {max_c_cost}")

    # Select best point using utility function on the valid set
    best = _select_by_utility(valid, cost_weight)

    # Use the stored leafs directly (no need to re-run optimization)
    return result.get_cut_circuit(best)


def _select_by_utility(frontier: list[CutPoint], cost_weight: float) -> CutPoint:
    """Select point from Pareto frontier using utility function.
    
    Minimizes: normalized_error + cost_weight * normalized_cost
    
    Both error and c_cost are normalized to [0, 1] using the frontier's
    own min/max values, so cost_weight is interpretable:
    - cost_weight=0: Only care about error
    - cost_weight=1: Equal weight (1% error reduction = 1% cost increase)
    - cost_weight=2: Cost is 2x as important as error
    
    Tiebreaker: when scores are equal, prefer lower quantum_cost (smaller subcircuits).
    
    Args:
        frontier: List of Pareto-optimal CutPoints.
        cost_weight: How much to weight classical cost (λ).
        
    Returns:
        The point that minimizes the utility function.
    """
    if len(frontier) == 1:
        return frontier[0]
    
    # Get normalization bounds from frontier
    min_error = min(p.max_error for p in frontier)
    max_error = max(p.max_error for p in frontier)
    min_cost = min(p.c_cost for p in frontier)
    max_cost = max(p.c_cost for p in frontier)
    
    error_range = max_error - min_error if max_error > min_error else 1
    cost_range = max_cost - min_cost if max_cost > min_cost else 1
    
    def score(p: CutPoint):
        # Normalize both to [0, 1]
        norm_error = (p.max_error - min_error) / error_range
        norm_cost = (p.c_cost - min_cost) / cost_range
        # Primary score: utility function
        # Tiebreaker: prefer smaller quantum_cost (add tiny fraction)
        return (norm_error + cost_weight * norm_cost, p.max_size)
    
    return min(frontier, key=score)
