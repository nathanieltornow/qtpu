"""Sweep compiler seeds and find the best seed per target Pareto max_size.

For each target max_size in SELECTED_MAX_SIZES, sweeps compiler seeds 0..N
and reports the seed with the lowest c_cost at that max_size on the
Pareto frontier. Uses lightweight metrics (c_cost, sampling_cost from
the CutPoint) to avoid compiling sub-circuits.

Usage
-----
    uv run python -m evaluation.hardware.search_seeds
    uv run python -m evaluation.hardware.search_seeds 100   # 100 seeds
"""
from __future__ import annotations

import sys
from time import perf_counter

from qtpu.compiler.opt import get_pareto_frontier
from evaluation.hardware.clifford_qnn import build_clifford_qnn_conjugated
from evaluation.hardware.run_pareto import (
    CIRCUIT_SIZE,
    SELECTED_MAX_SIZES,
    N_TRIALS,
    SEED,
)


def main():
    import math

    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    print(
        f"Seed sweep: {CIRCUIT_SIZE}q conjugated, seeds=0..{n_seeds-1}, "
        f"n_trials={N_TRIALS}, targets={SELECTED_MAX_SIZES}",
        flush=True,
    )

    qc = build_clifford_qnn_conjugated(CIRCUIT_SIZE, seed=SEED)

    # per-seed: target -> (c_cost, CutPoint) with exact max_size match, cheapest at that size
    per_seed: dict[int, dict[int, tuple[float, object]]] = {}
    # per-target global best (c_cost, seed, CutPoint)
    best: dict[int, tuple[float, int, object]] = {}

    t0 = perf_counter()
    for s in range(n_seeds):
        r = get_pareto_frontier(
            qc, max_sampling_cost=150, num_workers=1, n_trials=N_TRIALS, seed=s,
        )
        frontier = [p for p in r.pareto_frontier if p.max_size < CIRCUIT_SIZE]
        this_seed: dict[int, tuple[float, object]] = {}
        for target in SELECTED_MAX_SIZES:
            exact = [q for q in frontier if q.max_size == target]
            if not exact:
                continue
            p = min(exact, key=lambda q: q.c_cost)
            this_seed[target] = (p.c_cost, p)
            if target not in best or p.c_cost < best[target][0]:
                best[target] = (p.c_cost, s, p)
        per_seed[s] = this_seed

        if (s + 1) % 5 == 0:
            elapsed = perf_counter() - t0
            print(
                f"  [{s+1}/{n_seeds}] {elapsed:.0f}s elapsed, "
                f"hits={ {t: best[t][1] for t in best} }",
                flush=True,
            )

    print()
    print("=" * 72)
    print("PER-TARGET BEST SEED (cheapest c_cost at exact max_size)")
    print("=" * 72)
    for target in SELECTED_MAX_SIZES:
        if target not in best:
            print(f"  max_size={target:<3}  NO MATCH in any seed")
            continue
        c_cost, seed, p = best[target]
        print(
            f"  max_size={target:<3}  seed={seed:<3}  "
            f"c_cost={c_cost:.3e}  max_error={p.max_error:.4f}  "
            f"sampling_cost={p.sampling_cost:.2f}"
        )

    # Rank seeds whose single frontier covers ALL targets, by sum(log c_cost)
    # (geometric mean) — penalizes a seed being 10× worse at any one target.
    print()
    print("=" * 72)
    print("BEST SINGLE SEED (one frontier covering all targets, ranked by geomean c_cost)")
    print("=" * 72)
    full_cover = {
        s: d for s, d in per_seed.items()
        if all(t in d for t in SELECTED_MAX_SIZES)
    }
    if not full_cover:
        print(f"  No single seed has points at all of {SELECTED_MAX_SIZES}")
    else:
        ranked = sorted(
            full_cover.items(),
            key=lambda kv: sum(math.log(kv[1][t][0]) for t in SELECTED_MAX_SIZES),
        )
        header = "  seed  " + "  ".join(f"ms={t:<3}c_cost " for t in SELECTED_MAX_SIZES) + " geomean"
        print(header)
        for s, d in ranked[:10]:
            row = f"  {s:<4}  "
            costs = [d[t][0] for t in SELECTED_MAX_SIZES]
            row += "  ".join(f"      {c:.2e} " for c in costs)
            gmean = math.exp(sum(math.log(c) for c in costs) / len(costs))
            row += f"  {gmean:.2e}"
            print(row)
        print(f"\n  {len(full_cover)}/{n_seeds} seeds have full coverage")


if __name__ == "__main__":
    main()
