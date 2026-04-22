"""QNN fidelity + QPU-time validation sweep on IBM Marrakesh.

Fulfills revision Condition 2 + D3:
  - Workload: QNN at 20-80 qubits (`build_clifford_qnn_conjugated`, observed=None → Z^n)
  - Metric: retention (raw <P>), with cut-vs-mono ratio
  - D3: logs both estimated QPU time (FakeMarrakesh `estimate_duration` × shots)
    and the measured `actual_qpu_time` from job metrics, for each submission.

Usage:
    # Env-load first (IBM credentials), then run:
    set -a; source unlimited.env; set +a
    uv run python -m evaluation.hardware.sweep_marrakesh_qnn --pilot
    uv run python -m evaluation.hardware.sweep_marrakesh_qnn
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import torch

import qtpu
from qtpu.runtime import HEinsumRuntime
from qtpu.runtime.ibm_backend import (
    IBMBackend,
    _defer_qpd_measures,
    _strip_resets_and_measure,
)
from qtpu.transforms import circuit_to_heinsum
from evaluation.hardware.clifford_qnn import build_clifford_qnn_conjugated


try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass


BACKEND_NAME = "ibm_marrakesh"
QPU_SIZE = 10
LOG_PATH = "logs/hardware/qnn_marrakesh.jsonl"

PILOT = ([20], 5_000, 2_000)
FULL = ([10, 20, 30, 40, 50, 60, 80], 20_000, 10_000)


def _connect(backend_name: str):
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(
        channel="ibm_cloud",
        token=os.environ["IBM_TOKEN"],
        instance=os.environ["IBM_CRN"],
    )
    return service.backend(backend_name)


_FAKE_BACKEND = None

# IBM shot overhead: deterministic per-shot time IBM bills that `estimate_duration`
# doesn't include. Two components, both platform constants:
#   - `default_rep_delay` (250 µs on Marrakesh): inter-shot passive reset /
#     thermalization window, pulled from FakeMarrakesh.configuration().
#   - Control-plane wrapping (~90 µs empirical): classical overhead per shot
#     (job trampoline, readout classification, etc.) that's invariant in n.
# Together: ~340 µs/shot. Measured fit vs. IBM's billed quantum_seconds on
# n=20 mono (20k shots): 6µs circuit + 340µs × 20k = 6.9s, observed 7.0s.
_IBM_CONTROL_OVERHEAD_S = 90e-6


def _fake_backend():
    """Lazily load FakeMarrakesh (local, no network) for D3 duration estimates."""
    global _FAKE_BACKEND
    if _FAKE_BACKEND is None:
        from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
        _FAKE_BACKEND = FakeMarrakesh()
    return _FAKE_BACKEND


def _per_shot_overhead_s() -> float:
    """Deterministic per-shot overhead billed by IBM (rep_delay + control)."""
    fake = _fake_backend()
    rep_delay = float(fake.configuration().default_rep_delay)
    return rep_delay + _IBM_CONTROL_OVERHEAD_S


def _estimate_circuit_seconds(transpiled, backend=None):
    """Estimate per-shot quantum execution time (gate schedule only) via FakeMarrakesh.

    Returns just the circuit execution time — the time IBM's scheduler needs to
    play the pulses. The full billed QPU time per shot is this value plus the
    platform overhead from `_per_shot_overhead_s()`. Callers that want to match
    IBM's `quantum_seconds` metric must add the overhead term, multiplied by
    the total number of shots across all submitted circuits.
    """
    fake = _fake_backend()
    try:
        return float(transpiled.estimate_duration(fake.target, unit="s"))
    except Exception:
        return float("nan")


def _submit_monolithic(qc, device, shots: int) -> dict:
    from qiskit.compiler import transpile
    from qiskit_ibm_runtime import SamplerV2

    qc_m = qc.copy()
    _strip_resets_and_measure(qc_m)

    t0 = perf_counter()
    transpiled = transpile(qc_m, backend=device, optimization_level=3)
    compile_time = perf_counter() - t0

    depth = transpiled.depth()
    n_2q = sum(1 for i in transpiled.data if i.operation.num_qubits == 2)
    per_shot_s = _estimate_circuit_seconds(transpiled, device)
    # Full billed per-shot time = circuit pulses + rep_delay + control overhead.
    # Without the overhead term, the estimate is ~70x short of IBM's quantum_seconds.
    shot_overhead = _per_shot_overhead_s()
    estimated_qpu_time = (
        (per_shot_s + shot_overhead) * shots if per_shot_s == per_shot_s else float("nan")
    )

    sampler = SamplerV2(mode=device)
    t0 = perf_counter()
    job = sampler.run([transpiled], shots=shots)
    result = job.result()
    hw_wall = perf_counter() - t0

    actual_qpu = job.metrics().get("usage", {}).get("quantum_seconds", 0.0)
    counts = result[0].data.meas.get_counts()
    total = sum(counts.values())
    expval = sum(((-1) ** bs.count("1")) * c / total for bs, c in counts.items())

    return {
        "expval": expval,
        "transpiled_depth": depth,
        "transpiled_2q_gates": n_2q,
        "estimated_per_shot_s": per_shot_s,
        "estimated_qpu_time_s": estimated_qpu_time,
        "actual_qpu_time": actual_qpu,
        "compile_time": compile_time,
        "hw_wall_time": hw_wall,
        "num_flat_circuits": 1,
        "num_jobs": 1,
    }


def _submit_cut(qc, device, shots: int, qpu_size: int) -> dict:
    """Cut + reconstruct via HEinsumRuntime on IBM hardware.

    For D3: estimates per-fragment per-shot durations on FakeMarrakesh-compatible
    target, sums across flats × shots to get an estimated QPU time, and records
    the measured actual_qpu_time from job metrics.
    """
    from qiskit.compiler import transpile

    t0 = perf_counter()
    cut = qtpu.cut(qc, max_size=qpu_size, cost_weight=1000, n_trials=20, seed=1, num_workers=1)
    htn = circuit_to_heinsum(cut)
    compile_time = perf_counter() - t0

    n_flats = sum(len(list(qt.flat())) for qt in htn.quantum_tensors)

    # Pre-submission estimate: transpile ONE representative flat per partition
    # on FakeMarrakesh (local, fast) — all flats within a partition share gate
    # structure. Multiply by (flats per partition × shots). Much faster than
    # transpiling every flat on the live backend.
    fake = _fake_backend()
    shot_overhead = _per_shot_overhead_s()
    estimated_total = 0.0
    n_estimated = 0
    for qt in htn.quantum_tensors:
        flats = list(qt.flat())
        if not flats:
            continue
        # Find first non-trivial flat (has measurements after cleaning)
        sample = None
        for circ in flats:
            c = circ.decompose()
            c = _defer_qpd_measures(c)
            if not any(i.operation.name == "measure" for i in c.data):
                _strip_resets_and_measure(c)
            if any(i.operation.name == "measure" for i in c.data):
                sample = c
                break
        if sample is None:
            continue
        t = transpile(sample, backend=fake, optimization_level=3)
        d = _estimate_circuit_seconds(t)
        if d == d:
            # Each flat runs `shots` shots; each shot pays circuit time + platform overhead.
            estimated_total += (d + shot_overhead) * shots * len(flats)
            n_estimated += len(flats)

    backend = IBMBackend(backend=device, shots=shots, optimization_level=3)
    runtime = HEinsumRuntime(
        htn, backend=backend, dtype=torch.float64, device=torch.device("cpu")
    )
    runtime.prepare(optimize=False)

    t0 = perf_counter()
    res, _ = runtime.execute()
    hw_wall = perf_counter() - t0
    expval = float(res.item() if res.ndim == 0 else res.sum().item())

    return {
        "expval": expval,
        "num_partitions": len(htn.quantum_tensors),
        "num_flat_circuits": n_flats,
        "num_jobs": backend.total_jobs,
        "estimated_qpu_time_s": estimated_total,
        "estimated_flats_scored": n_estimated,
        "actual_qpu_time": backend.total_actual_qpu_time,
        "compile_time": compile_time,
        "hw_wall_time": hw_wall,
    }


def _append_log(entry: dict):
    path = Path(LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _already_completed_sizes(tier: str, shots_mono: int, shots_cut: int) -> set:
    """Return set of n that are already in the log with matching tier+shots."""
    p = Path(LOG_PATH)
    if not p.exists():
        return set()
    done = set()
    for line in p.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if (r.get("tier") == tier
                and r.get("shots_mono") == shots_mono
                and r.get("shots_cut") == shots_cut):
            done.add(r["n"])
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="Cheap pilot at small n, low shots")
    ap.add_argument("--resume", action="store_true",
                    help="Skip sizes already in log for this tier+shot config")
    args = ap.parse_args()

    sizes, shots_mono, shots_cut = PILOT if args.pilot else FULL
    tier = "pilot" if args.pilot else "full"

    if args.resume:
        done = _already_completed_sizes(tier, shots_mono, shots_cut)
        sizes = [n for n in sizes if n not in done]
        print(f"[resume] already completed: {sorted(done)}, running: {sizes}", flush=True)

    print(
        f"QNN Marrakesh sweep [{tier}]: qpu_size={QPU_SIZE}, "
        f"sizes={sizes}, shots_mono={shots_mono}, shots_cut={shots_cut}",
        flush=True,
    )

    device = _connect(BACKEND_NAME)
    status = device.status()
    print(
        f"Backend: {device.name} operational={status.operational} "
        f"pending={status.pending_jobs}",
        flush=True,
    )
    print(
        f"{'n':>3} | {'mono':>8} | {'cut':>8} {'parts':>5} {'flats':>5} | "
        f"{'ratio':>7} | {'m_est':>6} {'m_act':>6} | {'c_est':>6} {'c_act':>6}",
        flush=True,
    )
    print("-" * 92, flush=True)

    for n in sizes:
        # observed=None -> weight-n P_base = Z^n. Under QNN Clifford conjugation
        # this yields P_out with support ~0.65-0.80·n (verified n=20..80), so
        # every qubit actually carries observable weight and the mono-vs-cut
        # idle-noise asymmetry shows up in the signal. Previously used
        # observed=0, which under QNN's near-linear CX chain produced a
        # weight-2 P_out at every n — effectively a 2-qubit experiment.
        qc = build_clifford_qnn_conjugated(n, seed=42)

        mono = _submit_monolithic(qc, device, shots_mono)
        cut = _submit_cut(qc, device, shots_cut, QPU_SIZE)

        m = mono["expval"]
        c = cut["expval"]
        ratio = c / m if abs(m) > 0.05 else float("nan")

        row = {
            "tier": tier,
            "workload": "qnn",
            "n": n,
            "qpu_size": QPU_SIZE,
            "shots_mono": shots_mono,
            "shots_cut": shots_cut,
            "mono": mono,
            "cut": cut,
            "cut_mono_ratio": ratio,
        }
        _append_log(row)

        print(
            f"{n:>3} | {m:>+8.4f} | {c:>+8.4f} {cut['num_partitions']:>5} "
            f"{cut['num_flat_circuits']:>5} | {ratio:>7.3f} | "
            f"{mono['estimated_qpu_time_s']:>5.1f}s {mono['actual_qpu_time']:>5.1f}s | "
            f"{cut['estimated_qpu_time_s']:>5.1f}s {cut['actual_qpu_time']:>5.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
