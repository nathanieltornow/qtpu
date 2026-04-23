"""Explore the compiler's Pareto frontier of classical cost vs quantum error."""

from qiskit.circuit.random import random_circuit
from qtpu.compiler.opt import get_pareto_frontier


def main():
    # Build a 40-qubit circuit
    qc = random_circuit(40, depth=5, seed=42)

    # Get the full Pareto frontier
    result = get_pareto_frontier(qc, max_sampling_cost=100, n_trials=20, seed=0)

    print(f"Found {len(result.pareto_frontier)} Pareto-optimal solutions:\n")
    print(f"{'max_size':>10} {'c_cost':>12} {'max_error':>10}")
    print("-" * 35)
    for p in sorted(result.pareto_frontier, key=lambda p: p.c_cost):
        print(f"{p.max_size:>10} {p.c_cost:>12.0f} {p.max_error:>10.4f}")


if __name__ == "__main__":
    main()
