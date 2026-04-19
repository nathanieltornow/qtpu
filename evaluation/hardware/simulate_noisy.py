"""Noisy-simulator verification of the Pareto pipeline.

Runs the cut + HEinsum pipeline through a local noisy simulator that uses
FakeMarrakesh's noise model. Bridges the gap between the noiseless
verification (always +1.0) and the real-HW run (~0.0 everywhere) by
showing what the reconstruction returns under a realistic but controlled
noise model. Only small sizes / small max_size — large circuits are too
slow for density-matrix / statevector simulation with noise.

Usage
-----
    uv run python -m evaluation.hardware.simulate_noisy
    uv run python -m evaluation.hardware.simulate_noisy 20 10   # size 20, max_size 10
"""
from __future__ import annotations

import sys
from time import perf_counter

import numpy as np
import torch

import qtpu
from qtpu.runtime import HEinsumRuntime
from qtpu.runtime.backends import QuantumBackend
from qtpu.runtime.ibm_backend import _measure_non_reset_qubits
from qtpu.transforms import remove_operations_by_name
from evaluation.hardware.clifford_qnn import build_clifford_qnn_conjugated


class AerNoisyBackend(QuantumBackend):
    """Drop-in QuantumBackend that runs subcircuits on AerSimulator with
    FakeMarrakesh's noise model. Mirrors IBMBackend.evaluate but local."""

    def __init__(self, shots: int = 1000, seed: int = 0):
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel
        # FakeMarrakesh's noise model crashes Aer on macOS; use FakeBrisbane
        # as a noise proxy (same-vendor IBM device, comparable error rates).
        from qiskit_ibm_runtime.fake_provider import FakeBrisbane
        self._fake = FakeBrisbane()
        # Extract noise model only — we keep the circuit's logical qubit count
        # rather than expanding to the fake's 156 physical qubits (which would
        # crash Aer's density_matrix method).
        nm = NoiseModel.from_backend(self._fake)
        self._noise_model = nm
        self._basis_gates = nm.basis_gates
        self._sim = AerSimulator(
            noise_model=nm,
            basis_gates=nm.basis_gates,
        )
        self._shots = shots
        self._seed = seed

    @property
    def name(self) -> str:
        return "aer-noisy-marrakesh"

    def evaluate(self, qtensor, params, dtype, device):
        from qiskit.compiler import transpile

        t0 = perf_counter()
        flats = qtensor.flat()
        if not flats:
            return torch.zeros(qtensor.shape, dtype=dtype, device=device), 0.0, 0.0

        clean = []
        for circ in flats:
            c = circ.decompose()
            if c.parameters and params:
                pn = {p.name for p in c.parameters}
                to_bind = {k: v for k, v in params.items() if k in pn}
                if to_bind:
                    c = c.assign_parameters(to_bind)
            c = remove_operations_by_name(c, {"qpd_measure", "iswitch"}, inplace=False)
            if not any(i.operation.name == "measure" for i in c.data):
                _measure_non_reset_qubits(c)
            clean.append(c)

        # Transpile only to the noise-model's basis (no layout), so we stay
        # at logical-qubit count. Aer sometimes crashes on batched circuits
        # with this noise model; run one at a time to be safe.
        transpiled = transpile(clean, basis_gates=self._basis_gates, optimization_level=1)

        values = []
        for i, circ in enumerate(transpiled):
            result = self._sim.run(
                circ, shots=self._shots, seed_simulator=self._seed + i
            ).result()
            counts = result.get_counts()
            total = sum(counts.values())
            ev = sum(((-1) ** bs.replace(" ", "").count("1")) * c / total for bs, c in counts.items())
            values.append(ev)

        out = torch.tensor(values, dtype=dtype, device=device)
        if qtensor.shape:
            out = out.reshape(qtensor.shape)
        return out, perf_counter() - t0, 0.0


def run(n: int, qpu_size: int, shots: int = 1000) -> dict:
    qc = build_clifford_qnn_conjugated(n, seed=42)

    t0 = perf_counter()
    cut = qtpu.cut(
        qc, max_size=qpu_size, cost_weight=1000, n_trials=20, seed=1, num_workers=1
    )
    htn = qtpu.circuit_to_heinsum(cut)
    t_cut = perf_counter() - t0

    flats = [c.decompose() for qt in htn.quantum_tensors for c in qt.flat()]
    backend = AerNoisyBackend(shots=shots)
    rt = HEinsumRuntime(
        htn, backend=backend, dtype=torch.float64, device=torch.device("cpu")
    )
    rt.prepare(optimize=False)

    t0 = perf_counter()
    res, _ = rt.execute()
    t_exec = perf_counter() - t0
    val = float(res.item() if res.ndim == 0 else res.sum().item())

    return {
        "n": n,
        "qpu_size": qpu_size,
        "parts": len(htn.quantum_tensors),
        "flat_circs": len(flats),
        "shots": shots,
        "t_cut": t_cut,
        "t_exec": t_exec,
        "recon": val,
    }


def main():
    # Default sweep: small n, various qpu_size to see how the retention changes
    if len(sys.argv) == 3:
        sweep = [(int(sys.argv[1]), int(sys.argv[2]))]
    else:
        sweep = [
            (12, 6), (12, 10),
            (20, 6), (20, 10),
            (30, 6), (30, 10),
        ]

    print(f"Noisy-sim sweep (FakeMarrakesh noise model, shots=1000):", flush=True)
    print(f"  {'n':>3} {'QPU':>3} {'parts':>5} {'flats':>5}  "
          f"{'recon':>10}  {'retention':>9}  {'t_cut':>6} {'t_exec':>7}", flush=True)
    for n, qs in sweep:
        r = run(n, qs)
        retention = r["recon"] / 1.0  # ideal = +1
        print(
            f"  {r['n']:>3} {r['qpu_size']:>3} {r['parts']:>5} {r['flat_circs']:>5}  "
            f"{r['recon']:>+10.4f}  {retention:>+9.4f}  "
            f"{r['t_cut']:>5.1f}s {r['t_exec']:>6.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
