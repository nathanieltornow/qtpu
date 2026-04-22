"""Sim sweep of the Clifford-QNN benchmark beyond the hardware noise-floor ceiling.

Companion to `sweep_marrakesh_qnn.py`. Hardware on IBM Marrakesh caps out at
n=80 (both mono and cut enter the device noise floor there). This script runs
the same `build_clifford_qnn_conjugated` workload under the paper's uniform
Pauli depolarizing noise model (ε_cx=1e-2, ε_id=1e-3) at sizes up to n=150 so
the fidelity trend can be reported beyond what hardware can measure — the
fallback contingency explicitly named in the revision plan.

All-Clifford circuit + Pauli noise → AerSimulator stabilizer method handles
n=150 in seconds. Infrastructure reused from sweep_spreading.py:
AerStabilizerNoisy (cut backend), _insert_layered_idles (mono idle padding),
_make_noisy_sim (mono sim + noise).

Usage:
    uv run python -m evaluation.hardware.sim_sweep_qnn
"""
from __future__ import annotations

import qiskit_aer  # noqa: F401  -- must precede torch/qtpu (see sweep_spreading.py note)
from qiskit_aer import AerSimulator  # noqa: F401
from qiskit_aer.noise import NoiseModel, depolarizing_error  # noqa: F401

import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

import qtpu
from qtpu.runtime import HEinsumRuntime
from qtpu.runtime.ibm_backend import _strip_resets_and_measure
from evaluation.hardware.clifford_qnn import build_clifford_qnn_conjugated
from evaluation.hardware.sweep_spreading import (
    AerStabilizerNoisy, _make_noisy_sim, _insert_layered_idles,
)
from qiskit.circuit import QuantumCircuit
from qtpu.runtime.ibm_backend import _defer_qpd_measures
import math


# QPD basis-prep for the QNN's weight-n observable produces X/Y/Z per-qubit
# rotations (H, S\u2020H, I) as `u(\u03b8, \u03c6, \u03bb)` gates. Plus our conjugated-observable
# construction emits an X flip for sign=-1 branches and resets for I qubits.
# The spreading-benchmark rewriter only handles H and identity; extend to
# every Clifford `u` shape we expect to see in a QNN flat.
_TWO_PI = 2 * math.pi


def _mod2pi_close(a, b, tol=1e-8):
    d = (a - b) % _TWO_PI
    return d < tol or (_TWO_PI - d) < tol


def _rewrite_u_to_clifford_qnn(circ):
    """Map every 1q `u(\u03b8, \u03c6, \u03bb)` and `p(\u03b8)` in `circ` to Clifford gates.

    QPD basis-prep + opt_level=3 produces a long tail of Clifford 1q shapes
    (I, X, Y, Z, H, S, S\u2020, HS, SH, \u2026). Rather than enumerate them, we build a
    single-qubit sub-circuit, ask Qiskit for the equivalent Clifford object,
    and re-emit it as a clean Clifford decomposition (h/s/sdg/x/y/z) that
    stabilizer sim accepts. Non-Clifford ops raise here.
    """
    from qiskit.quantum_info import Clifford

    new = QuantumCircuit(*circ.qregs, *circ.cregs, name=circ.name)
    for instr in circ.data:
        op = instr.operation
        if op.name not in ("u", "p") or op.num_qubits != 1:
            new.append(op, instr.qubits, instr.clbits)
            continue
        q = instr.qubits[0]
        tmp = QuantumCircuit(1)
        tmp.append(op, [0])
        try:
            cliff = Clifford.from_circuit(tmp)
        except Exception as e:
            raise ValueError(
                f"Non-Clifford 1q op {op.name}{list(op.params)}: {e}"
            )
        decomp = cliff.to_circuit()
        for sub in decomp.data:
            new.append(sub.operation, [q], [])
    return new


class AerStabilizerNoisyQNN(AerStabilizerNoisy):
    """Same as AerStabilizerNoisy but uses the QNN-compatible u-rewriter."""

    def evaluate(self, qtensor, params, dtype, device):
        from time import perf_counter
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
            r = self._sim.run(c, shots=self._shots, seed_simulator=seed_ctr).result().get_counts()
            seed_ctr += 1
            t = sum(r.values())
            vals.append(sum(((-1) ** bs.replace(" ", "").count("1")) * v / t for bs, v in r.items()))
        out = torch.tensor(vals, dtype=dtype, device=device)
        if qtensor.shape:
            out = out.reshape(qtensor.shape)
        return out, perf_counter() - t0, 0.0


SIZES = [20, 40, 60, 80, 100, 120, 150]
SHOTS_MONO = 50_000
SHOTS_CUT = 10_000
QPU_SIZE = 10
MONO_SEEDS = [0, 1, 2, 3, 4]
CUT_SEED = 1
LOG_PATH = Path(os.path.expanduser("~/qtpu/logs/hardware/qnn_sim_sweep.jsonl"))


def mono_exp(qc, shots, seed):
    sim, _ = _make_noisy_sim()
    qc_m = qc.copy()
    _strip_resets_and_measure(qc_m)
    qc_m = _insert_layered_idles(qc_m)
    r = sim.run(qc_m, shots=shots, seed_simulator=seed).result().get_counts()
    t = sum(r.values())
    return sum(((-1) ** bs.replace(" ", "").count("1")) * c / t for bs, c in r.items())


def cut_exp(qc, qpu_size, shots, cut_seed):
    cut = qtpu.cut(qc, max_size=qpu_size, cost_weight=1000, n_trials=20, seed=cut_seed, num_workers=1)
    htn = qtpu.circuit_to_heinsum(cut)
    n_flats = sum(len(list(qt.flat())) for qt in htn.quantum_tensors)
    rt = HEinsumRuntime(
        htn, backend=AerStabilizerNoisyQNN(shots=shots),
        dtype=torch.float64, device=torch.device("cpu"),
    )
    rt.prepare(optimize=False)
    res, _ = rt.execute()
    val = float(res.item() if res.ndim == 0 else res.sum().item())
    return val, len(htn.quantum_tensors), n_flats


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    print(
        f"QNN sim sweep | sizes={SIZES} | qpu_size={QPU_SIZE} | "
        f"shots_mono={SHOTS_MONO} | shots_cut={SHOTS_CUT} | "
        f"mono_seeds={MONO_SEEDS} | cut_seed={CUT_SEED}",
        flush=True,
    )
    print(
        f"{'n':>3} | {'mono_mean':>9} \u00b1 {'std':>6} | {'cut':>8} | "
        f"{'ratio':>6} | {'parts':>5} {'flats':>5} | {'t':>6}",
        flush=True,
    )
    print("-" * 85, flush=True)

    for n in SIZES:
        qc = build_clifford_qnn_conjugated(n, seed=42)
        t0 = perf_counter()
        mono_vals = [mono_exp(qc, SHOTS_MONO, seed=s) for s in MONO_SEEDS]
        cut_val, parts, flats = cut_exp(qc, QPU_SIZE, SHOTS_CUT, cut_seed=CUT_SEED)
        dt = perf_counter() - t0

        m_mean, m_std = float(np.mean(mono_vals)), float(np.std(mono_vals))
        ratio = cut_val / m_mean if abs(m_mean) > 0.02 else float("nan")

        row = {
            "n": n, "qpu_size": QPU_SIZE,
            "shots_mono": SHOTS_MONO, "shots_cut": SHOTS_CUT,
            "mono_seeds": MONO_SEEDS, "cut_seed": CUT_SEED,
            "mono_vals": mono_vals, "mono_mean": m_mean, "mono_std": m_std,
            "cut": cut_val, "parts": parts, "flats": flats,
            "ratio": ratio, "elapsed_s": dt,
        }
        with LOG_PATH.open("a") as f:
            f.write(json.dumps(row) + "\n")

        print(
            f"{n:>3} | {m_mean:>+9.4f} \u00b1 {m_std:>6.4f} | {cut_val:>+8.4f} | "
            f"{ratio:>6.3f} | {parts:>5} {flats:>5} | {dt:>5.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
