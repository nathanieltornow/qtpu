"""Spreading-chain sweep: idle-gate noise on H^n + forward-CX chain.

Workload is all-Clifford, so we use AerSimulator(method='stabilizer') with a
Pauli depolarizing noise model on both `cx` (ε_cx ≈ 1e-2) and `id` (ε_id ≈
1e-3). The `id` channel is the key ingredient — it makes idle qubits a
first-class noise source, which is the mechanism by which cutting wins:

    mono:  O(n²) idle slots  (n qubits × ~n layers)
    cut:   O(n·W) idle slots  (K = n/W fragments × W² each)

So the mono retention falls like exp(-a·n²·ε_id) while cut retention only
falls like exp(-a·n·W·ε_id). At large n the gap grows linearly with n.

Usage:
    uv run python -m evaluation.hardware.sweep_spreading
"""
from __future__ import annotations

# Import qiskit-aer FIRST, before torch/qtpu. There's a nasty OpenMP/BLAS
# interaction (probably MKL vs OpenBLAS) that segfaults the stabilizer sim
# if torch is loaded before aer. Observed on macOS with uv-managed env.
import qiskit_aer  # noqa: F401
from qiskit_aer import AerSimulator  # noqa: F401
from qiskit_aer.noise import NoiseModel, depolarizing_error  # noqa: F401

import math
from time import perf_counter

import torch

import qtpu
from qtpu.runtime import HEinsumRuntime
from qtpu.runtime.backends import QuantumBackend
from qtpu.runtime.ibm_backend import _defer_qpd_measures, _strip_resets_and_measure
from evaluation.hardware.clifford_qnn import build_spreading_conjugated


EPS_CX = 1e-2
EPS_ID = 1e-3


# Layered idle insertion. Walk circuit layers via the DAG; for each layer,
# emit the layer's gates verbatim, then an `id` on every qubit that didn't
# participate in the layer (skipping classical-only ops and `id`s themselves).
# Called AFTER cutting for cut fragments (so each fragment only pays its own
# depth-W idle cost) and BEFORE running for mono (so mono pays its full
# depth-n idle cost). `measure`/`reset`/`barrier` layers are treated as
# "no idle inserted" — they represent end-of-circuit bookkeeping, not
# decoherence-accumulating idle time on the other qubits.
def _insert_layered_idles(qc):
    from qiskit.circuit import QuantumCircuit
    from qiskit.converters import circuit_to_dag
    dag = circuit_to_dag(qc)
    new = QuantumCircuit(*qc.qregs, *qc.cregs, name=qc.name)
    all_qubits = list(qc.qubits)
    for layer in dag.layers():
        layer_graph = layer["graph"]
        ops = list(layer_graph.op_nodes())
        # Passthrough layer: nothing we want idle-padded after it.
        is_terminal = any(
            node.op.name in ("measure", "reset", "barrier") for node in ops
        )
        touched = set()
        for node in ops:
            new.append(node.op, node.qargs, node.cargs)
            for q in node.qargs:
                touched.add(q)
        if is_terminal:
            continue
        for q in all_qubits:
            if q not in touched:
                new.id(q)
    return new


# QPD basis-prep for Pauli measurement fragments emits exactly one non-trivial
# 1q rotation shape: U(π/2, 0, π) = H (up to global phase). Detect that exact
# Clifford angle and rewrite in place rather than adding a transpile pass that
# might prune `id` gates. Any other `u` param would be a bug in our assumption
# and we raise loudly.
def _rewrite_u_to_clifford(circ):
    from qiskit.circuit import QuantumCircuit
    import math
    new = QuantumCircuit(*circ.qregs, *circ.cregs, name=circ.name)
    for instr in circ.data:
        op = instr.operation
        if op.name != "u":
            new.append(op, instr.qubits, instr.clbits)
            continue
        theta, phi, lam = (float(p) for p in op.params)
        def _close(a, b):
            return abs((a - b) % (2 * math.pi)) < 1e-9 or abs(((a - b) % (2 * math.pi)) - 2 * math.pi) < 1e-9
        if _close(theta, math.pi / 2) and _close(phi, 0.0) and _close(lam, math.pi):
            new.h(instr.qubits[0])
        elif _close(theta, 0.0) and _close(phi, 0.0) and _close(lam, 0.0):
            new.id(instr.qubits[0])
        else:
            raise ValueError(
                f"Unexpected non-Clifford u gate in QPD fragment: "
                f"theta={theta}, phi={phi}, lam={lam}"
            )
    return new


def _make_noisy_sim():
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(EPS_CX, 2), ["cx"])
    nm.add_all_qubit_quantum_error(depolarizing_error(EPS_ID, 1), ["id"])
    sim = AerSimulator(method="stabilizer", noise_model=nm)
    return sim, nm


class AerStabilizerNoisy(QuantumBackend):
    """Stabilizer + Pauli depolarizing noise backend for cut subcircuits.

    Decomposes each flat subcircuit, defers qpd_measures to ancillas, strips
    terminal resets, and measures the remainder. No transpile pass — all
    gates are already Clifford (h, s, sdg, cx, id, x, y, z) and stabilizer
    method handles them directly. Skipping transpile is critical: any
    optimization pass will happily collapse `id` chains and destroy the
    idle-noise signal we built into the circuit.
    """

    def __init__(self, shots: int):
        self._sim, _ = _make_noisy_sim()
        self._shots = shots

    @property
    def name(self):
        return "aer-stabilizer-noisy"

    def evaluate(self, qtensor, params, dtype, device):
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
            c = _rewrite_u_to_clifford(c)
            c = _insert_layered_idles(c)
            r = self._sim.run(c, shots=self._shots, seed_simulator=seed_ctr).result().get_counts()
            seed_ctr += 1
            t = sum(r.values())
            vals.append(sum(((-1) ** bs.replace(" ", "").count("1")) * v / t for bs, v in r.items()))
        out = torch.tensor(vals, dtype=dtype, device=device)
        if qtensor.shape:
            out = out.reshape(qtensor.shape)
        return out, perf_counter() - t0, 0.0


def mono_run(qc, shots: int) -> float:
    sim, _ = _make_noisy_sim()
    qc_m = qc.copy()
    _strip_resets_and_measure(qc_m)
    qc_m = _insert_layered_idles(qc_m)
    r = sim.run(qc_m, shots=shots, seed_simulator=0).result().get_counts()
    total = sum(r.values())
    return sum(((-1) ** bs.replace(" ", "").count("1")) * c / total for bs, c in r.items())


def cut_run(qc, qpu_size: int, shots: int):
    cut = qtpu.cut(qc, max_size=qpu_size, cost_weight=1000, n_trials=20, seed=1, num_workers=1)
    htn = qtpu.circuit_to_heinsum(cut)
    n_flats = sum(len(list(qt.flat())) for qt in htn.quantum_tensors)
    rt = HEinsumRuntime(
        htn,
        backend=AerStabilizerNoisy(shots=shots),
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    rt.prepare(optimize=False)
    res, _ = rt.execute()
    val = float(res.item() if res.ndim == 0 else res.sum().item())
    return val, len(htn.quantum_tensors), n_flats


def main():
    qpu_size = 6
    shots_mono = 20_000
    shots_cut = 5_000
    sizes = [10, 20, 30, 40, 60]

    print(
        f"Spreading sweep: qpu_size={qpu_size}, observable=Z_0, "
        f"ε_cx={EPS_CX}, ε_id={EPS_ID}, shots_mono={shots_mono}, shots_cut={shots_cut}"
    )
    print("Ideal ⟨P⟩ = +1. Mono retention should decay faster than cut retention as n grows.\n")
    print(
        f"{'n':>3} | {'mono':>8} | {'cut':>8} {'parts':>5} {'flats':>5} | "
        f"{'cut/mono':>8} | {'t_m':>6} {'t_c':>6}"
    )
    print("-" * 75)

    for n in sizes:
        qc = build_spreading_conjugated(n, observed=0)

        t0 = perf_counter()
        m = mono_run(qc, shots_mono)
        t_m = perf_counter() - t0

        t0 = perf_counter()
        c, parts, flats = cut_run(qc, qpu_size, shots_cut)
        t_c = perf_counter() - t0

        ratio = c / m if abs(m) > 0.05 else float("nan")
        print(
            f"{n:>3} | {m:>+8.4f} | {c:>+8.4f} {parts:>5} {flats:>5} | "
            f"{ratio:>8.4f} | {t_m:>5.1f}s {t_c:>5.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
