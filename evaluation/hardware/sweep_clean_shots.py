"""Pure-shot-noise sweep: AerSimulator with NO noise model.

Isolates the question "does the QPD reconstruction pipeline return the
right answer under finite sampling alone?" by stripping gate noise entirely.
If recon stays ≈ +1 within √(K/N) for all n, the pipeline is shot-noise-only-
limited and unbiased; any degradation on the noisy sweep is then purely from
gate/readout noise, not from QPD sum variance amplification.

Usage:
    uv run python -m evaluation.hardware.sweep_clean_shots
"""
from __future__ import annotations

import math
from time import perf_counter

import torch

import qtpu
from qtpu.runtime import HEinsumRuntime
from qtpu.runtime.backends import QuantumBackend
from qtpu.runtime.ibm_backend import _defer_qpd_measures, _strip_resets_and_measure
from evaluation.hardware.clifford_qnn import build_clifford_qnn_conjugated


class AerClean(QuantumBackend):
    """AerSimulator with no noise model — pure shot sampling. Seed offset
    lets us run multiple independent trials."""
    def __init__(self, shots: int, seed_off: int = 0):
        from qiskit_aer import AerSimulator
        self._sim = AerSimulator()
        self._shots = shots
        self._seed_off = seed_off

    @property
    def name(self):
        return "aer-clean"

    def evaluate(self, qtensor, params, dtype, device):
        t0 = perf_counter()
        flats = qtensor.flat()
        if not flats:
            return torch.zeros(qtensor.shape, dtype=dtype, device=device), 0.0, 0.0
        vals = []
        for i, circ in enumerate(flats):
            c = circ.decompose()
            if c.parameters and params:
                pn = {p.name for p in c.parameters}
                to_bind = {k: v for k, v in params.items() if k in pn}
                if to_bind:
                    c = c.assign_parameters(to_bind)
            c = _defer_qpd_measures(c)
            if not any(ii.operation.name == "measure" for ii in c.data):
                _strip_resets_and_measure(c)
            seed = i + self._seed_off * 100_000
            r = self._sim.run(c, shots=self._shots, seed_simulator=seed).result().get_counts()
            t = sum(r.values())
            vals.append(sum(((-1) ** bs.replace(" ", "").count("1")) * v / t for bs, v in r.items()))
        out = torch.tensor(vals, dtype=dtype, device=device)
        if qtensor.shape:
            out = out.reshape(qtensor.shape)
        return out, perf_counter() - t0, 0.0


def mono_clean(qc, shots: int, seed: int) -> float:
    from qiskit_aer import AerSimulator
    sim = AerSimulator()
    qc_m = qc.copy()
    _strip_resets_and_measure(qc_m)
    r = sim.run(qc_m, shots=shots, seed_simulator=seed).result().get_counts()
    total = sum(r.values())
    return sum(((-1) ** bs.replace(" ", "").count("1")) * c / total for bs, c in r.items())


def cut_clean(qc, qpu_size: int, shots: int, seed_off: int):
    cut = qtpu.cut(qc, max_size=qpu_size, cost_weight=1000, n_trials=20, seed=1, num_workers=1)
    htn = qtpu.circuit_to_heinsum(cut)
    n_flats = sum(len(list(qt.flat())) for qt in htn.quantum_tensors)
    rt = HEinsumRuntime(htn, backend=AerClean(shots=shots, seed_off=seed_off),
                        dtype=torch.float64, device=torch.device("cpu"))
    rt.prepare(optimize=False)
    res, _ = rt.execute()
    val = float(res.item() if res.ndim == 0 else res.sum().item())
    return val, len(htn.quantum_tensors), n_flats


def stats(vals):
    mean = sum(vals) / len(vals)
    std = math.sqrt(sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1))
    return mean, std


def main():
    qpu_size = 6
    shots = 10_000
    n_trials = 5
    sizes = [8, 12, 16, 20, 24]

    print(f"Clean shot-noise sweep: qpu_size={qpu_size}, shots={shots}, trials={n_trials}")
    print("Ideal ⟨P⟩ = +1.000 everywhere. Deviation = sampling variance only.\n")
    print(f"{'n':>3} | {'mono':>10} {'m_std':>7} | "
          f"{'cut':>10} {'c_std':>7} {'SEM_th':>7} | {'parts':>5} {'flats':>5}")
    print("-" * 80)

    for n in sizes:
        qc = build_clifford_qnn_conjugated(n, seed=42)

        mono_vals = [mono_clean(qc, shots, seed=s) for s in range(n_trials)]
        mono_mean, mono_std = stats(mono_vals)

        cut_vals = []
        parts = flats = 0
        for trial in range(n_trials):
            v, parts, flats = cut_clean(qc, qpu_size, shots, seed_off=trial)
            cut_vals.append(v)
        cut_mean, cut_std = stats(cut_vals)
        cut_sem_th = math.sqrt(flats / shots)

        print(f"{n:>3} | {mono_mean:>+10.4f} {mono_std:>7.4f} | "
              f"{cut_mean:>+10.4f} {cut_std:>7.4f} {cut_sem_th:>7.4f} | "
              f"{parts:>5} {flats:>5}", flush=True)


if __name__ == "__main__":
    main()
