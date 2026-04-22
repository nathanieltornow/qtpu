"""Pauli-noise simulation of the 40q Pareto frontier.

Companion to `run_pareto.py` (real hardware on Marrakesh) and `sim_sweep_qnn.py`
(Pauli-noise sim of the scale sweep). Runs the same selected Pareto-frontier
points (max_size ∈ {3, 5, 10, 15, 20}) + monolithic under a uniform depolarizing
noise model (ε_cx=1e-2, ε_id=1e-3) — stabilizer method, 100k shots per flat so
QPD reconstruction variance is negligible.

The hypothesis being tested: on HW the Pareto frontier shows a fidelity peak at
max_size=10 (sampling variance eats gains past that). Under idealized Pauli
noise with enough shots, the frontier should be monotone (fidelity climbs with
c_cost). This separates "compiler cost-model prediction" from
"reconstruction-variance confound".

Usage:
    uv run python -m evaluation.hardware.sim_pareto_40q
"""
from __future__ import annotations

import qiskit_aer  # noqa: F401  -- must precede torch/qtpu
from qiskit_aer import AerSimulator  # noqa: F401
from qiskit_aer.noise import NoiseModel, depolarizing_error  # noqa: F401

import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from qtpu.compiler.opt import get_pareto_frontier, CutPoint
from qtpu.runtime import HEinsumRuntime
from qtpu.runtime.ibm_backend import _strip_resets_and_measure
from qtpu.transforms import circuit_to_heinsum

from evaluation.hardware.clifford_qnn import build_clifford_qnn_conjugated
from evaluation.hardware.sweep_spreading import (
    _make_noisy_sim, _insert_layered_idles,
)
from evaluation.hardware.sim_sweep_qnn import AerStabilizerNoisyQNN


CIRCUIT_SIZE = 80
SHOTS_CUT = 20_000
SHOTS_MONO = 20_000
N_TRIALS = 20
SEED = 42              # circuit builder seed
COMPILER_SEED = 1      # Pareto enumeration seed
EVAL_SEEDS = [0, 1, 2]  # re-evaluate each point under multiple sim seeds
SELECTED_MAX_SIZES = [3, 5, 10, 15, 20, 30, 40, 60]
LOG_PATH = Path("logs/hardware/sim_pareto_80q.jsonl")


def select_frontier_subset(frontier, max_sizes):
    if not frontier:
        return []
    cheapest_by_size = {}
    for p in frontier:
        cur = cheapest_by_size.get(p.max_size)
        if cur is None or p.c_cost < cur.c_cost:
            cheapest_by_size[p.max_size] = p
    available = sorted(cheapest_by_size.keys())
    picked = {}
    for target in max_sizes:
        nearest = min(available, key=lambda s: (abs(s - target), s))
        picked[nearest] = cheapest_by_size[nearest]
    return [picked[s] for s in sorted(picked.keys())]


def mono_exp(qc, shots, seed):
    sim, _ = _make_noisy_sim()
    qc_m = qc.copy()
    _strip_resets_and_measure(qc_m)
    qc_m = _insert_layered_idles(qc_m)
    r = sim.run(qc_m, shots=shots, seed_simulator=seed).result().get_counts()
    t = sum(r.values())
    return sum(((-1) ** bs.replace(" ", "").count("1")) * c / t for bs, c in r.items())


def cut_exp(cut_circuit, shots, seed):
    """Run a pre-cut circuit through the Pauli-noise backend (sim_seed controls shot noise)."""
    htn = circuit_to_heinsum(cut_circuit)
    n_flats = sum(len(list(qt.flat())) for qt in htn.quantum_tensors)
    backend = AerStabilizerNoisyQNN(shots=shots)
    # Reseed the simulator run via a per-run counter — backend doesn't take seed directly,
    # but it threads seed_ctr through each flat; we nudge by monkey-patching _shots unused
    # and rely on AerStabilizerNoisyQNN's internal seed_ctr (starts at 0 each evaluate()).
    # For seed variation we just rebuild the backend — seed_ctr restarts but AerSimulator
    # stabilizer counts are deterministic given seed_simulator. To vary, patch seed offset:
    backend._seed_offset = seed * 10_000
    # Inline patch: replicate AerStabilizerNoisyQNN.evaluate but with seed offset
    rt = HEinsumRuntime(
        htn, backend=_SeededBackend(shots=shots, seed_offset=seed * 10_000),
        dtype=torch.float64, device=torch.device("cpu"),
    )
    rt.prepare(optimize=True)
    res, _ = rt.execute()
    val = float(res.item() if res.ndim == 0 else res.sum().item())
    return val, len(htn.quantum_tensors), n_flats


class _SeededBackend(AerStabilizerNoisyQNN):
    """AerStabilizerNoisyQNN with a per-instance seed offset for sim variance."""

    def __init__(self, shots, seed_offset=0):
        super().__init__(shots=shots)
        self._seed_offset = seed_offset

    def evaluate(self, qtensor, params, dtype, device):
        from qtpu.runtime.ibm_backend import _defer_qpd_measures
        from evaluation.hardware.sim_sweep_qnn import _rewrite_u_to_clifford_qnn

        t0 = perf_counter()
        flats = qtensor.flat()
        if not flats:
            return torch.zeros(qtensor.shape, dtype=dtype, device=device), 0.0, 0.0
        vals = []
        seed_ctr = 0
        for circ in flats:
            c = circ.decompose()
            if c.parameters and params:
                pn = {p.name for p in c.parameters}
                to_bind = {k: v for k, v in params.items() if k in pn}
                if to_bind:
                    c = c.assign_parameters(to_bind)
            c = _defer_qpd_measures(c)
            if not any(ii.operation.name == "measure" for ii in c.data):
                _strip_resets_and_measure(c)
            if not any(ii.operation.name == "measure" for ii in c.data):
                vals.append(1.0)
                continue
            c = _rewrite_u_to_clifford_qnn(c)
            c = _insert_layered_idles(c)
            seed_sim = self._seed_offset + seed_ctr
            r = self._sim.run(c, shots=self._shots, seed_simulator=seed_sim).result().get_counts()
            seed_ctr += 1
            t = sum(r.values())
            vals.append(sum(((-1) ** bs.replace(" ", "").count("1")) * v / t for bs, v in r.items()))
        out = torch.tensor(vals, dtype=dtype, device=device)
        if qtensor.shape:
            out = out.reshape(qtensor.shape)
        return out, perf_counter() - t0, 0.0


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    print(
        f"Pareto sim sweep @ {CIRCUIT_SIZE}q Clifford-QNN | "
        f"shots_cut={SHOTS_CUT} shots_mono={SHOTS_MONO} seeds={EVAL_SEEDS}",
        flush=True,
    )

    qc = build_clifford_qnn_conjugated(CIRCUIT_SIZE, seed=SEED)

    t0 = perf_counter()
    opt_result = get_pareto_frontier(
        qc, max_sampling_cost=500, num_workers=1,
        n_trials=N_TRIALS, seed=COMPILER_SEED,
    )
    full_frontier = sorted(
        (p for p in opt_result.pareto_frontier if p.max_size < CIRCUIT_SIZE),
        key=lambda p: p.max_size,
    )
    frontier = select_frontier_subset(full_frontier, SELECTED_MAX_SIZES)
    print(f"Pareto frontier: {len(full_frontier)} cuts in {perf_counter()-t0:.1f}s; "
          f"running {len(frontier)} subset + monolithic", flush=True)
    for p in frontier:
        print(f"  max_size={p.max_size:3d} c_cost={p.c_cost:8.1f} "
              f"max_error={p.max_error:.4f} sampling_cost={p.sampling_cost:.1f}", flush=True)

    print(f"\n{'mode':>12} {'max_sz':>6} {'c_cost':>8} | "
          f"{'mean':>9} ± {'std':>6} | {'parts':>5} {'flats':>5} | {'t':>6}", flush=True)
    print("-" * 85, flush=True)

    # Run each Pareto point across seeds
    for point in frontier:
        cut_circuit = opt_result.get_cut_circuit(point)
        t0 = perf_counter()
        vals = []
        parts, flats = None, None
        for s in EVAL_SEEDS:
            v, p_, f_ = cut_exp(cut_circuit, SHOTS_CUT, seed=s)
            vals.append(v)
            parts, flats = p_, f_
        dt = perf_counter() - t0
        mean, std = float(np.mean(vals)), float(np.std(vals))

        row = {
            "mode": "cut",
            "max_size": point.max_size,
            "c_cost": point.c_cost,
            "max_error_estimate": point.max_error,
            "sampling_cost": point.sampling_cost,
            "shots": SHOTS_CUT,
            "seeds": EVAL_SEEDS,
            "vals": vals,
            "fidelity_mean": mean,
            "fidelity_std": std,
            "num_partitions": parts,
            "num_flat_circuits": flats,
            "elapsed_s": dt,
        }
        with LOG_PATH.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{'cut':>12} {point.max_size:>6} {point.c_cost:>8.1f} | "
              f"{mean:>+9.4f} ± {std:>6.4f} | {parts:>5} {flats:>5} | {dt:>5.1f}s", flush=True)

    # Monolithic
    t0 = perf_counter()
    vals = [mono_exp(qc, SHOTS_MONO, seed=s) for s in EVAL_SEEDS]
    dt = perf_counter() - t0
    mean, std = float(np.mean(vals)), float(np.std(vals))
    row = {
        "mode": "monolithic",
        "max_size": CIRCUIT_SIZE,
        "c_cost": 0.0,
        "max_error_estimate": None,
        "sampling_cost": 0.0,
        "shots": SHOTS_MONO,
        "seeds": EVAL_SEEDS,
        "vals": vals,
        "fidelity_mean": mean,
        "fidelity_std": std,
        "num_partitions": 1,
        "num_flat_circuits": 1,
        "elapsed_s": dt,
    }
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"{'monolithic':>12} {CIRCUIT_SIZE:>6} {0.0:>8.1f} | "
          f"{mean:>+9.4f} ± {std:>6.4f} | {1:>5} {1:>5} | {dt:>5.1f}s", flush=True)


if __name__ == "__main__":
    main()
