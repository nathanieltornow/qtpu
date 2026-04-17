"""Validate the analytical Mitiq baseline projection.

The end-to-end baseline in `run.py` computes:
    n_total_circuits = n_batch * n_support * n_qac_experiments * n_zne

This script actually wires in real Mitiq (mitiq.zne.execute_with_zne) on a
cut 20-qubit QNN and counts how many circuits Mitiq really submits. If the
counts match, the analytical projection is correct and we can safely keep
using it at larger sizes where end-to-end execution is infeasible.

Usage:
    uv run python -m evaluation.use_cases.end_to_end.validate_mitiq
"""

from time import perf_counter

import numpy as np
from mitiq.zne import execute_with_zne
from mitiq.zne.inference import RichardsonFactory
from mqt.bench import get_benchmark_indep
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import SparsePauliOp
from qiskit_addon_cutting import (
    expand_observables,
    generate_cutting_experiments,
    partition_problem,
)

import qtpu

CIRCUIT_SIZE = 20
QPU_SIZE = 10
BATCH_SIZE = 5       # small for the PoC
NUM_SUPPORT = 3
ZNE_SCALE_FACTORS = [1.0, 3.0, 5.0]


def defer_measurements(circuit: QuantumCircuit) -> QuantumCircuit:
    """Defer all mid-circuit measurements to the end via ancilla qubits.

    For each measure(q, c), allocate a fresh ancilla, replace the measurement
    with cx(q, ancilla), and append measure(ancilla, c) after all gates.
    Mirrors the pattern in `qtpu.transforms._defer_circuit` used by
    `decompose_qpd_measures(defer=True)`, but targets regular `measure` ops
    (QAC has already expanded `qpd_measure` into concrete measurements by the
    time we receive its experiments).

    The resulting circuit has a pure unitary block followed by terminal
    measurements, which Mitiq's folding tolerates.
    """
    measure_targets = [
        (instr.qubits[0], instr.clbits[0])
        for instr in circuit.data
        if instr.operation.name == "measure"
    ]
    n = len(measure_targets)

    new = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    if n == 0:
        for instr in circuit.data:
            new.append(instr.operation, instr.qubits, instr.clbits)
        return new

    ancillas = QuantumRegister(n, name="deferred_ancilla")
    new.add_register(ancillas)

    k = 0
    for instr in circuit.data:
        if instr.operation.name == "measure":
            new.cx(instr.qubits[0], ancillas[k])
            k += 1
        else:
            new.append(instr.operation, instr.qubits, instr.clbits)

    for k, (_q, c) in enumerate(measure_targets):
        new.measure(ancillas[k], c)

    return new


def make_counting_executor():
    """Return (executor_fn, counter_list). Mitiq requires a typed function,
    not a callable class — it introspects return annotations and classes
    don't survive its unwrapping."""
    counter = {"n": 0, "circuits": []}

    def executor(circuit) -> float:
        counter["n"] += 1
        counter["circuits"].append(circuit)
        return 0.0  # placeholder expectation value

    return executor, counter


def main():
    print("=" * 70)
    print(f"Mitiq projection validator — {CIRCUIT_SIZE}q QNN, {QPU_SIZE}q QPU")
    print(f"Batch={BATCH_SIZE}, Support={NUM_SUPPORT}, ZNE scales={ZNE_SCALE_FACTORS}")
    print("=" * 70)

    # --- Step 1: QNN + qTPU cutting (same as baseline) ---
    qc = get_benchmark_indep("qnn", circuit_size=CIRCUIT_SIZE, opt_level=3)
    qc = qc.remove_final_measurements(inplace=False)

    t0 = perf_counter()
    cut_circuit = qtpu.cut(
        qc, max_size=QPU_SIZE, cost_weight=1000, n_trials=20, seed=42
    )
    t_cut = perf_counter() - t0

    # --- Step 2: QAC generate experiments ---
    observable = SparsePauliOp(["Z" * qc.num_qubits])
    obs_expanded = expand_observables(observable.paulis, qc, cut_circuit)
    partitioned = partition_problem(circuit=cut_circuit, observables=obs_expanded)

    t0 = perf_counter()
    subexperiments, _ = generate_cutting_experiments(
        circuits=partitioned.subcircuits,
        observables=partitioned.subobservables,
        num_samples=np.inf,
    )
    t_qac = perf_counter() - t0

    all_experiments = [exp for exps in subexperiments.values() for exp in exps]
    n_qac = len(all_experiments)
    n_zne = len(ZNE_SCALE_FACTORS)
    print(f"\n[qTPU cut]  time={t_cut:.2f}s  num_partitions={len(partitioned.subcircuits)}")
    print(f"[QAC gen ]  time={t_qac:.2f}s  n_qac_experiments={n_qac}")

    # --- Step 3: defer all mid-circuit measurements to the end via ancillas.
    # QAC experiments have `observable_measurements` and `qpd_measurements`
    # classical registers with mid-circuit measures; Mitiq's folding only
    # tolerates terminal measurements. Deferring preserves outcome semantics
    # (unlike stripping), and makes the pre-measurement block a pure unitary
    # that Mitiq can fold. ---
    deferred = [defer_measurements(exp) for exp in all_experiments]

    # --- Step 4: run Mitiq's execute_with_zne on each experiment ---
    # We fix one (batch, support) pair — the batch/support axis is analytical
    # multiplication for the baseline and Mitiq doesn't interact with it.
    factory = RichardsonFactory(scale_factors=ZNE_SCALE_FACTORS)
    executor, counter = make_counting_executor()

    t0 = perf_counter()
    for i, exp in enumerate(deferred):
        before = counter["n"]
        _ = execute_with_zne(exp, executor, factory=factory)
        if i == 0:
            print(f"\n[Mitiq ]   first experiment: "
                  f"Mitiq called executor {counter['n'] - before} times")
    t_mitiq = perf_counter() - t0
    n_mitiq_per_bs = counter["n"]  # total for one (b,s) pair

    # --- Step 5: totals ---
    total_mitiq = BATCH_SIZE * NUM_SUPPORT * n_mitiq_per_bs
    total_analytical = BATCH_SIZE * NUM_SUPPORT * n_qac * n_zne

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Mitiq actual   (one batch,support): {n_mitiq_per_bs:>10d}")
    print(f"Analytical     (one batch,support): {n_qac * n_zne:>10d}"
          f"    (= n_qac={n_qac} × n_zne={n_zne})")
    print(f"Mitiq total    ({BATCH_SIZE}×{NUM_SUPPORT}): {total_mitiq:>10d}")
    print(f"Analytical tot ({BATCH_SIZE}×{NUM_SUPPORT}): {total_analytical:>10d}")
    print(f"Ratio:                              {total_mitiq / total_analytical:.3f}")
    print(f"Mitiq runtime (counting only):      {t_mitiq:.2f}s")

    if total_mitiq == total_analytical:
        print("\n✓ PROJECTION VALIDATED — counts match exactly.")
        print("  We can safely use the analytical projection at larger sizes.")
    else:
        delta = total_mitiq - total_analytical
        print(f"\n✗ MISMATCH — Δ = {delta:+d} circuits ({delta / total_analytical * 100:+.2f}%)")
        print("  Investigate how Mitiq counts circuits vs our formula.")


if __name__ == "__main__":
    main()
