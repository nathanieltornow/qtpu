from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING

from qtpu.transforms import circuit_to_heinsum
from qtpu.core.heinsum import HEinsum

from ._opt import get_pareto_frontier, CutPoint

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


@dataclass
class HEinsumCutPoint:
    """A single point in the HEinsum optimization space."""

    c_cost: float  # Total classical cost (sum across all subcircuits)
    max_error: float  # Max error across all subcircuits
    max_size: int  # Max subcircuit width across all subcircuits
    # Per-tensor cut points (for reconstruction)
    tensor_cuts: list[CutPoint]


@dataclass
class HEinsumOptimizationResult:
    """Result of HEinsum optimization with combined Pareto frontier."""

    # All combinations explored
    all_points: list[HEinsumCutPoint]

    # Pareto-optimal points (c_cost vs max_error)
    pareto_frontier: list[HEinsumCutPoint]

    # Original HEinsum (for reconstruction)
    original_heinsum: HEinsum

    # Per-tensor optimization results
    tensor_results: list[tuple[int, any]]  # (qt_index, OptimizationResult)

    def filter(
        self,
        max_size: int | None = None,
        max_c_cost: float | None = None,
        max_error: float | None = None,
    ) -> list[HEinsumCutPoint]:
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
    ) -> HEinsumCutPoint | None:
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

        def score(p: HEinsumCutPoint):
            norm_cost = (p.c_cost - min_cost) / cost_range
            norm_error = (p.max_error - min_error) / error_range
            return norm_error + cost_weight * norm_cost

        return min(valid, key=score)

    def get_heinsum(self, point: HEinsumCutPoint) -> HEinsum:
        """Reconstruct HEinsum from a cut point.

        Args:
            point: A HEinsumCutPoint from the frontier.

        Returns:
            The optimized HEinsum with cut subcircuits.
        """
        all_qts = []
        all_cts = self.original_heinsum.classical_tensors.copy()

        for (qt_idx, opt_result), cut_point in zip(
            self.tensor_results, point.tensor_cuts
        ):
            cut_circ = opt_result.get_cut_circuit(cut_point)
            opt_heinsum = circuit_to_heinsum(cut_circ)
            all_qts.extend(opt_heinsum.quantum_tensors)
            all_cts.extend(opt_heinsum.classical_tensors)

        return HEinsum(
            qtensors=all_qts,
            ctensors=all_cts,
            input_tensors=self.original_heinsum.input_tensors,
            output_inds=self.original_heinsum.output_inds,
        )

    def get_all_heinsums(
        self,
        max_size: int | None = None,
        max_c_cost: float | None = None,
        max_error: float | None = None,
    ) -> list[tuple[HEinsumCutPoint, HEinsum]]:
        """Get all HEinsums from the Pareto frontier.

        Args:
            max_size: Filter by max subcircuit width.
            max_c_cost: Filter by max classical cost.
            max_error: Filter by max error.

        Returns:
            List of (HEinsumCutPoint, HEinsum) tuples.
        """
        valid = self.filter(
            max_size=max_size, max_c_cost=max_c_cost, max_error=max_error
        )
        return [(p, self.get_heinsum(p)) for p in valid]


def _compute_pareto_frontier(points: list[HEinsumCutPoint]) -> list[HEinsumCutPoint]:
    """Compute the Pareto frontier on c_cost vs max_error."""
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


@dataclass
class OptimizationParameters:
    num_workers: int | None = None
    n_trials: int = 100
    seed: int | None = None


def optimize(
    heinsum: HEinsum, params: OptimizationParameters | None = None
) -> HEinsumOptimizationResult:
    """Optimize a HEinsum by finding the Pareto frontier of cut combinations.

    This function computes the Pareto frontier over all combinations of cuts
    for each quantum tensor. The combined metrics are:
    - c_cost: Sum of c_costs across all subcircuits (total classical work)
    - max_error: Max of max_errors across subcircuits (bottleneck error)

    Args:
        heinsum: The original HEinsum specification.
        params: Optimization parameters.

    Returns:
        HEinsumOptimizationResult with the combined Pareto frontier.

    Example:
        result = optimize(heinsum)

        # Get all Pareto-optimal HEinsums
        for point, opt_heinsum in result.get_all_heinsums():
            print(f"c_cost={point.c_cost:.2e}, error={point.max_error:.3f}")

        # Select best with constraint
        best = result.select_best(max_size=10, cost_weight=1.0)
        opt_heinsum = result.get_heinsum(best)
    """
    if params is None:
        params = OptimizationParameters()

    # Get Pareto frontier for each quantum tensor
    # Skip tensors that already have ISwitches (already cut) - they have non-empty shape
    tensor_results: list[tuple[int, any]] = []

    for i, qt in enumerate(heinsum.quantum_tensors):
        # Check if this tensor is already cut (has ISwitches)
        if qt.shape:
            # Already has ISwitches - create a dummy "no further cut" result
            # with a single point representing the current state
            error = sum(
                0.01 if inst.operation.num_qubits == 2 else 0.001
                for inst in qt.circuit.data
            )
            dummy_point = CutPoint(
                c_cost=0,
                max_error=error,
                max_size=qt.circuit.num_qubits,
                sampling_cost=0,
                leafs=None,  # No cut needed
            )
            # Create a minimal result that just returns the original
            from qtpu.transforms import remove_operations_by_name

            clean_circuit = remove_operations_by_name(
                qt.circuit, {"barrier"}, inplace=False
            )

            class DummyOptResult:
                pareto_frontier = [dummy_point]
                ir = None
                _circuit = clean_circuit

                def get_cut_circuit(self, point):
                    return self._circuit

            tensor_results.append((i, DummyOptResult()))
        else:
            # Raw circuit - optimize it
            opt_result = get_pareto_frontier(
                qt.circuit,
                num_workers=params.num_workers,
                n_trials=params.n_trials,
                seed=params.seed,
            )
            tensor_results.append((i, opt_result))

    # Combine all Pareto frontiers
    # For N tensors with frontiers of sizes [k1, k2, ..., kN],
    # we get k1 * k2 * ... * kN combinations
    frontiers = [result.pareto_frontier for _, result in tensor_results]

    if not frontiers:
        # No quantum tensors - return empty result
        return HEinsumOptimizationResult(
            all_points=[],
            pareto_frontier=[],
            original_heinsum=heinsum,
            tensor_results=tensor_results,
        )

    # Generate all combinations
    all_combinations: list[HEinsumCutPoint] = []

    for combo in product(*frontiers):
        # combo is a tuple of CutPoints, one per tensor
        combined_c_cost = sum(p.c_cost for p in combo)
        combined_max_error = max(p.max_error for p in combo)
        combined_max_size = max(p.max_size for p in combo)

        all_combinations.append(
            HEinsumCutPoint(
                c_cost=combined_c_cost,
                max_error=combined_max_error,
                max_size=combined_max_size,
                tensor_cuts=list(combo),
            )
        )

    # Compute combined Pareto frontier
    pareto = _compute_pareto_frontier(all_combinations)

    return HEinsumOptimizationResult(
        all_points=all_combinations,
        pareto_frontier=pareto,
        original_heinsum=heinsum,
        tensor_results=tensor_results,
    )
