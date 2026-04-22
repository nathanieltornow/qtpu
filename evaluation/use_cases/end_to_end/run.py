"""End-to-End Composability Benchmark (OSDI Condition 1)
======================================================

Demonstrates that qTPU can express an error-mitigated, scalable hybrid ML
workload as a single hTN, and measures end-to-end performance vs. the
baseline pipeline (QAC + Mitiq-style ZNE + manual batching).

Workload:
  QNN circuits (20–60 qubits) with:
    - Circuit cutting to fit a 10-qubit QPU
    - ZNE error mitigation (noise levels 1, 3, 5)
    - Hybrid ML batch inference over feature vectors

qTPU approach:
  1. Build QNN HEinsum with batch/support ISwitches
  2. Cut to 10-qubit subcircuits via qtpu.cut()
  3. Add ZNE ISwitches + Richardson extrapolation CTensors
  4. Single HEinsum → compile once → execute

Baseline approach:
  1. QAC for cutting
  2. Mitiq-style explicit ZNE circuit enumeration
  3. Manual batching loop
"""

from __future__ import annotations

import sys
import tracemalloc
from time import perf_counter

import numpy as np
import torch
from qiskit.circuit import Parameter, QuantumCircuit, ClassicalRegister

import benchkit as bk

import qtpu
from qtpu.core import HEinsum, QuantumTensor, CTensor, ISwitch, TensorSpec
from qtpu.compiler.codegen import quantum_tensor_to_cudaq
from qtpu.runtime.baseline import run_heinsum, run_batch, estimate_qpu_time
from evaluation.analysis import estimate_runtime


# =============================================================================
# Configuration
# =============================================================================

CIRCUIT_SIZES = [20, 30, 40, 50, 60]
QPU_SIZE = 10
BATCH_SIZE = 50
FEATURE_DIM = 4
NUM_SUPPORT = 20
NUM_LAYERS = 2
ZNE_NOISE_LEVELS = [1, 3, 5]
MAX_ZNE_GATES = 5  # ZNE ISwitches per subcircuit (limits tensor shape explosion)


# =============================================================================
# Circuit Construction
# =============================================================================


def create_feature_map(
    num_qubits: int, x: np.ndarray, layers: int = 2
) -> QuantumCircuit:
    """Hardware-efficient feature map encoding data into qubits."""
    qc = QuantumCircuit(num_qubits)
    num_features = len(x)
    for layer in range(layers):
        for i in range(num_qubits):
            qc.h(i)
        for i in range(num_qubits):
            qc.ry(float(x[i % num_features]), i)
        for i in range(num_qubits - 1):
            qc.cz(i, i + 1)
    return qc


class _BatchSelector:
    """Picklable ISwitch selector for batch feature maps."""

    def __init__(self, num_qubits: int, X: np.ndarray, layers: int, inverse: bool = False):
        self.num_qubits = num_qubits
        self.X = X
        self.layers = layers
        self.inverse = inverse

    def __call__(self, idx: int) -> QuantumCircuit:
        fm = create_feature_map(self.num_qubits, self.X[idx], self.layers)
        return fm.inverse() if self.inverse else fm


def create_zne_iswitch(
    base_gate: QuantumCircuit, noise_levels: list[int], idx: str
) -> ISwitch:
    """Create ISwitch for ZNE noise folding."""
    param = Parameter(idx)

    def selector(level_idx: int) -> QuantumCircuit:
        level = noise_levels[level_idx]
        folded = QuantumCircuit(base_gate.num_qubits)
        folded.compose(base_gate, inplace=True)
        for _ in range((level - 1) // 2):
            folded.compose(base_gate.inverse(), inplace=True)
            folded.compose(base_gate, inplace=True)
        return folded

    return ISwitch(param, base_gate.num_qubits, len(noise_levels), selector)


def zne_coefficients(noise_levels: list[int]) -> np.ndarray:
    """Richardson extrapolation coefficients for ZNE."""
    V = np.vander(noise_levels, increasing=True)
    return np.linalg.inv(V)[0]


# =============================================================================
# qTPU End-to-End Pipeline
# =============================================================================


def build_e2e_qtpu(
    num_qubits: int,
    X_batch: np.ndarray,
    X_support: np.ndarray,
    W: np.ndarray,
    max_subcircuit_size: int,
    noise_levels: list[int],
) -> tuple[HEinsum, float]:
    """Build end-to-end qTPU HEinsum: cutting + ZNE + ML batching.

    Pipeline:
      1. Build a plain QNN circuit (no ISwitches)
      2. Cut it to fit QPU via qtpu.cut()
      3. Convert to HEinsum (produces subcircuits + QPD tensors)
      4. Add ZNE ISwitches to subcircuit single-qubit gates
      5. Add batch/support ISwitches for ML inference
      6. Compose everything into one HEinsum

    Returns:
        Tuple of (heinsum, compile_time).
    """
    from mqt.bench import get_benchmark_indep

    n_batch = len(X_batch)
    n_support = len(X_support)

    # --- Step 1: Build a plain QNN circuit (no ISwitches) ---
    qc = get_benchmark_indep("qnn", circuit_size=num_qubits, opt_level=3)
    qc = qc.remove_final_measurements(inplace=False)

    # --- Step 2: Cut to fit QPU ---
    compile_start = perf_counter()
    cut_circuit = qtpu.cut(
        qc, max_size=max_subcircuit_size, cost_weight=1000,
        n_trials=20, seed=42,
    )

    # --- Step 3: Convert to HEinsum ---
    htn = qtpu.circuit_to_heinsum(cut_circuit)

    # --- Step 4: Add ZNE ISwitches to single-qubit gates in each subcircuit ---
    zne_coeffs = zne_coefficients(noise_levels)
    zne_ctensors = []
    enriched_qtensors = []

    for qt_idx, qt in enumerate(htn.quantum_tensors):
        circuit = qt.circuit
        new_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)
        zne_count = 0

        for instr in circuit.data:
            op = instr.operation
            if isinstance(op, ISwitch):
                new_circuit.append(op, instr.qubits, instr.clbits)
            elif (
                zne_count < MAX_ZNE_GATES
                and op.num_qubits == 1
                and op.name not in (
                    "measure",
                    "barrier",
                    "reset",
                    "qpd_measure",
                )
            ):
                base_gate = QuantumCircuit(1)
                base_gate.append(op, [0])
                zne_idx = f"zne_{qt_idx}_{zne_count}"
                iswitch = create_zne_iswitch(base_gate, noise_levels, zne_idx)
                new_circuit.append(iswitch, instr.qubits)
                zne_ctensors.append(CTensor(zne_coeffs, (zne_idx,)))
                zne_count += 1
            else:
                new_circuit.append(op, instr.qubits, instr.clbits)

        enriched_qtensors.append(QuantumTensor(new_circuit))

    # --- Step 5: Add ML batch/support ISwitches as a separate quantum tensor ---
    ml_circuit = QuantumCircuit(num_qubits)
    ml_circuit.add_register(ClassicalRegister(num_qubits))

    batch_param = Parameter("batch")
    support_param = Parameter("support")

    batch_iswitch = ISwitch(
        batch_param, num_qubits, n_batch,
        _BatchSelector(num_qubits, X_batch, NUM_LAYERS, inverse=False),
    )
    support_iswitch = ISwitch(
        support_param, num_qubits, n_support,
        _BatchSelector(num_qubits, X_support, NUM_LAYERS, inverse=True),
    )

    ml_circuit.append(batch_iswitch, range(num_qubits))
    ml_circuit.append(support_iswitch, range(num_qubits))
    ml_circuit.measure(range(num_qubits), range(num_qubits))

    ml_qtensor = QuantumTensor(ml_circuit)

    # --- Step 6: Compose into single HEinsum ---
    # Weight tensor for ML kernel
    W_tensor = CTensor(torch.tensor(W, dtype=torch.float64), inds=("support",))

    all_qtensors = enriched_qtensors + [ml_qtensor]
    all_ctensors = list(htn.classical_tensors) + zne_ctensors + [W_tensor]

    heinsum = HEinsum(
        qtensors=all_qtensors,
        ctensors=all_ctensors,
        input_tensors=list(htn.input_tensors),
        output_inds=("batch",),
    )

    compile_time = perf_counter() - compile_start
    return heinsum, compile_time


# =============================================================================
# Baseline End-to-End Pipeline
# =============================================================================


def run_e2e_baseline(
    num_qubits: int,
    X_batch: np.ndarray,
    X_support: np.ndarray,
    W: np.ndarray,
    max_subcircuit_size: int,
    noise_levels: list[int],
    num_zne_gates: int,
) -> dict:
    """Run baseline pipeline: QAC cutting + real Mitiq ZNE + manual batch loop.

    This is the "three separate tools" baseline reviewers asked for:
      (1) qiskit-addon-cutting (QAC) for cutting + basis enumeration
      (2) mitiq.zne.execute_with_zne for noise folding
      (3) a manual Python loop over (batch, support) pairs

    Mitiq is invoked on every QAC experiment with a counting executor; the
    quantum cost is then estimated from one representative × the measured
    Mitiq call count. `validate_mitiq.py` independently confirms that the
    Mitiq-call formula (n_qac × n_zne) is exact (ratio 1.000).
    """
    from mqt.bench import get_benchmark_indep
    from qiskit_addon_cutting import (
        expand_observables,
        generate_cutting_experiments,
        partition_problem,
    )
    from qiskit.quantum_info import SparsePauliOp
    from mitiq.zne import execute_with_zne
    from mitiq.zne.inference import RichardsonFactory
    from evaluation.use_cases.end_to_end.validate_mitiq import defer_measurements

    n_batch = len(X_batch)
    n_support = len(X_support)
    n_zne = len(noise_levels)

    # --- Step 1: cut (shared compiler — same cut quality for both paths) ---
    qc = get_benchmark_indep("qnn", circuit_size=num_qubits, opt_level=3)
    qc_orig = qc.remove_final_measurements(inplace=False)

    compile_start = perf_counter()
    cut_circuit = qtpu.cut(
        qc_orig, max_size=max_subcircuit_size, cost_weight=1000,
        n_trials=20, seed=42,
    )

    # --- Step 2: QAC experiment generation ---
    observable = SparsePauliOp(["Z" * qc_orig.num_qubits])
    observables_expanded = expand_observables(
        observable.paulis, qc_orig, cut_circuit
    )
    partitioned = partition_problem(
        circuit=cut_circuit, observables=observables_expanded
    )
    subexperiments, coefficients = generate_cutting_experiments(
        circuits=partitioned.subcircuits,
        observables=partitioned.subobservables,
        num_samples=np.inf,
    )
    compile_time = perf_counter() - compile_start

    all_experiments = [exp for exps in subexperiments.values() for exp in exps]
    n_qac_experiments = len(all_experiments)

    # --- Step 3: defer mid-circuit measurements so Mitiq can fold ---
    # QAC emits circuits with mid-circuit `measure` ops on observable_measurements
    # and qpd_measurements classical registers. Mitiq's folding only tolerates
    # terminal measurements; deferring preserves outcome semantics (unlike
    # stripping) and gives Mitiq a pure unitary block to fold over.
    deferred_experiments = [defer_measurements(e) for e in all_experiments]

    # --- Step 4: run Mitiq on every QAC experiment ---
    # NOTE: executor has no return annotation. Mitiq's Executor compares the
    # `return` annotation by identity against a list of types; under PEP 563
    # (from __future__ import annotations) a `-> float` annotation becomes
    # the string 'float' and fails the check. No annotation => Mitiq assumes
    # float-returning, which is what we want.
    mitiq_calls = {"n": 0}

    def mitiq_executor(circuit):
        mitiq_calls["n"] += 1
        return 0.0  # placeholder — we're counting, not evaluating

    factory = RichardsonFactory(scale_factors=[float(n) for n in noise_levels])

    mitiq_start = perf_counter()
    for exp in deferred_experiments:
        _ = execute_with_zne(exp, mitiq_executor, factory=factory)
    calls_per_experiment = mitiq_calls["n"] / max(n_qac_experiments, 1)
    mitiq_time = perf_counter() - mitiq_start

    # --- Step 5: classical reconstruction — real Mitiq + QAC + W math ---
    # Per (batch, support) pair a user would run:
    #   (a) Richardson-extrapolate n_zne expvals per QAC experiment → scalar
    #   (b) linearly combine n_qac extrapolated expvals via QAC coefficients
    #   (c) weight the per-(b,s) scalars by W[s] and reduce over support
    # We execute this arithmetic on dummy scalars (np.random). Classical cost
    # is structural (shape-bound), not data-dependent — symmetric with what
    # the qTPU path does via `tree.contract(arrays)` on dummy arrays.
    #
    # Richardson coefficients at scale=0 are the first row of the inverse
    # Vandermonde over the ZNE scale factors; this is exactly the linear
    # combination Mitiq's RichardsonFactory.reduce() applies internally.
    # validate_mitiq.py established ratio=1.000 between Mitiq's actual
    # circuit count and the n_qac × n_zne analytical formula that assumes
    # this same reduction, so using the analytical coefficients here is
    # correctness-equivalent to per-experiment RichardsonFactory calls but
    # permits numpy vectorisation over the 14M+ reductions at 50q.
    richardson_coeffs = zne_coefficients(noise_levels).astype(np.float64)
    qac_coeffs = np.ones(n_qac_experiments, dtype=np.float64)
    W_arr = np.asarray(W, dtype=np.float64)

    # Per-batch peak allocation: (n_support × n_qac × n_zne) float64.
    # Cap at ~1 GB to bound memory; fall back to subset-projection above it.
    elems_per_batch = n_support * n_qac_experiments * n_zne
    PER_BATCH_ELEM_CAP = 128 * 1024 * 1024  # ~1 GB of float64

    if n_qac_experiments == 0:
        classical_time = 0.0
        classical_time_source = "measured"
    elif elems_per_batch > PER_BATCH_ELEM_CAP:
        # Full allocation is prohibitive: measure one (b, s) slice's
        # Richardson+QAC reduction and project linearly to n_batch × n_support.
        # The per-batch W-reduction is O(n_support), negligible vs. the inner.
        classical_time_source = "projected_from_subset"
        rng = np.random.default_rng(0)
        classical_start = perf_counter()
        expvals_one = rng.standard_normal((n_qac_experiments, n_zne))
        zne_reduced_one = expvals_one @ richardson_coeffs   # (n_qac,)
        _ = float(zne_reduced_one @ qac_coeffs)             # (scalar)
        subset_time = perf_counter() - classical_start
        classical_time = subset_time * n_batch * n_support
    else:
        classical_time_source = "measured"
        rng = np.random.default_rng(0)
        classical_start = perf_counter()
        per_batch = np.empty(n_batch, dtype=np.float64)
        for b in range(n_batch):
            # shape (n_support, n_qac, n_zne) — one dummy expval per circuit
            expvals = rng.standard_normal((n_support, n_qac_experiments, n_zne))
            # (a) Richardson reduction over ZNE axis
            zne_reduced = expvals @ richardson_coeffs           # (n_support, n_qac)
            # (b) QAC coefficient combination over experiments
            qac_reduced = zne_reduced @ qac_coeffs              # (n_support,)
            # (c) W-weighted reduction over support → per-batch scalar
            per_batch[b] = qac_reduced @ W_arr
        _ = float(per_batch.sum())
        classical_time = perf_counter() - classical_start

    # --- Step 6: totals ---
    total_circuits = n_batch * n_support * n_qac_experiments * int(calls_per_experiment)

    # QPU time estimate: one representative × total count.
    if all_experiments:
        rep_time = estimate_runtime(all_experiments[:1])
        total_quantum_time = rep_time * total_circuits
    else:
        total_quantum_time = 0.0

    # Generated code size: each unique circuit becomes one generated kernel;
    # in the baseline pipeline every (b, s, experiment, zne_level) tuple is
    # a distinct circuit the user has to emit. Measure one representative via
    # the same codegen path qTPU uses (quantum_tensor_to_cudaq) and multiply.
    if all_experiments:
        rep_qt = QuantumTensor(deferred_experiments[0])
        _, rep_lines = quantum_tensor_to_cudaq(
            rep_qt.circuit, rep_qt.shape, kernel_name="baseline_kernel"
        )
        total_code_lines = rep_lines * total_circuits
    else:
        total_code_lines = 0

    return {
        "compile_time": compile_time,
        "quantum_time": total_quantum_time,
        "classical_time": classical_time,
        "classical_time_source": classical_time_source,
        "mitiq_time": mitiq_time,
        "num_circuits": total_circuits,
        "num_qac_experiments": n_qac_experiments,
        "mitiq_calls_per_experiment": int(calls_per_experiment),
        "total_code_lines": total_code_lines,
    }


# =============================================================================
# Benchmark Functions
# =============================================================================


@bk.foreach(circuit_size=CIRCUIT_SIZES)
@bk.log("logs/end_to_end/qtpu.jsonl")
def bench_qtpu(circuit_size: int) -> dict | None:
    """Benchmark qTPU end-to-end pipeline."""
    print(f"qTPU E2E: qubits={circuit_size}")

    np.random.seed(42)
    X_batch = np.random.randn(BATCH_SIZE, FEATURE_DIM) * np.pi
    X_support = np.random.randn(NUM_SUPPORT, FEATURE_DIM) * np.pi
    W = np.random.randn(NUM_SUPPORT) * 0.1

    try:
        import math

        start = perf_counter()
        heinsum, compile_time = build_e2e_qtpu(
            circuit_size, X_batch, X_support, W, QPU_SIZE, ZNE_NOISE_LEVELS
        )

        # Compute metrics directly (don't use run_heinsum — the ZNE ISwitches
        # create exponentially many flat circuits that can't be enumerated)

        # QPU time: estimate from one representative subcircuit per qtensor,
        # then multiply by total flat count
        total_flat_circuits = 0
        est_qpu_time = 0.0
        for qt in heinsum.quantum_tensors:
            n_flat = math.prod(qt.shape) if qt.shape else 1
            total_flat_circuits += n_flat
            # Estimate from first flat circuit only
            rep_circuits = qt.flat()[:1]
            if rep_circuits:
                rep_time = estimate_runtime(
                    [c.decompose() for c in rep_circuits]
                )
                est_qpu_time += rep_time * n_flat

        # Classical contraction cost
        tree, arrays = heinsum.to_dummy_tn()
        if tree is not None:
            contract_start = perf_counter()
            tree.contract(arrays)
            classical_time = perf_counter() - contract_start
        else:
            classical_time = 0.0

        # Code lines: one kernel per qtensor (qTPU compiles once per qtensor)
        total_code_lines = 0
        for qt in heinsum.quantum_tensors:
            _, num_lines = quantum_tensor_to_cudaq(
                qt.circuit, qt.shape, kernel_name="e2e_kernel"
            )
            total_code_lines += num_lines

        total_time = perf_counter() - start

        return {
            "compile_time": compile_time,
            "quantum_time": est_qpu_time,
            "classical_time": classical_time,
            "total_time": total_time,
            "num_circuits": total_flat_circuits,
            "num_subcircuits": len(heinsum.quantum_tensors),
            "num_ctensors": len(heinsum.classical_tensors),
            "total_code_lines": total_code_lines,
        }
    except Exception as e:
        print(f"  Error: {e}")
        import traceback

        traceback.print_exc()
        return None


@bk.foreach(circuit_size=CIRCUIT_SIZES)
@bk.log("logs/end_to_end/baseline.jsonl")
def bench_baseline(circuit_size: int) -> dict | None:
    """Benchmark baseline pipeline (QAC experiments + Mitiq ZNE + manual batching)."""
    print(f"Baseline E2E: qubits={circuit_size}")

    np.random.seed(42)
    X_batch = np.random.randn(BATCH_SIZE, FEATURE_DIM) * np.pi
    X_support = np.random.randn(NUM_SUPPORT, FEATURE_DIM) * np.pi
    W = np.random.randn(NUM_SUPPORT) * 0.1

    try:
        start = perf_counter()
        result = run_e2e_baseline(
            circuit_size,
            X_batch,
            X_support,
            W,
            QPU_SIZE,
            ZNE_NOISE_LEVELS,
            num_zne_gates=MAX_ZNE_GATES,
        )
        total_time = perf_counter() - start

        return {
            "compile_time": result["compile_time"],
            "quantum_time": result["quantum_time"],
            "classical_time": result["classical_time"],
            "classical_time_source": result["classical_time_source"],
            "mitiq_time": result["mitiq_time"],
            "total_time": total_time,
            "num_circuits": result["num_circuits"],
            "num_qac_experiments": result["num_qac_experiments"],
            "mitiq_calls_per_experiment": result["mitiq_calls_per_experiment"],
            "total_code_lines": result["total_code_lines"],
        }
    except Exception as e:
        print(f"  Error: {e}")
        import traceback

        traceback.print_exc()
        return None


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    usage = """
End-to-End Composability Benchmark (OSDI Condition 1)

Usage: python -m evaluation.use_cases.end_to_end.run <command>

Commands:
    qtpu        Run qTPU end-to-end pipeline
    baseline    Run baseline pipeline (QAC + Mitiq + manual batching)
    all         Run both

Configuration:
    Circuit sizes: {CIRCUIT_SIZES}
    QPU size:      {QPU_SIZE} qubits
    Batch size:    {BATCH_SIZE}
    Feature dim:   {FEATURE_DIM}
    Support vecs:  {NUM_SUPPORT}
    ZNE levels:    {ZNE_NOISE_LEVELS}
""".format(
        CIRCUIT_SIZES=CIRCUIT_SIZES,
        QPU_SIZE=QPU_SIZE,
        BATCH_SIZE=BATCH_SIZE,
        FEATURE_DIM=FEATURE_DIM,
        NUM_SUPPORT=NUM_SUPPORT,
        ZNE_NOISE_LEVELS=ZNE_NOISE_LEVELS,
    )

    if len(sys.argv) < 2:
        print(usage)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "qtpu":
        bench_qtpu()
    elif cmd == "baseline":
        bench_baseline()
    elif cmd == "all":
        print("Running all end-to-end benchmarks...")
        bench_qtpu()
        bench_baseline()
    elif cmd == "quick":
        # Quick test with smallest circuit
        print("Quick test: 20-qubit QNN")
        np.random.seed(42)
        X_batch = np.random.randn(10, FEATURE_DIM) * np.pi
        X_support = np.random.randn(5, FEATURE_DIM) * np.pi
        W = np.random.randn(5) * 0.1

        print("\nqTPU pipeline:")
        heinsum, ct = build_e2e_qtpu(20, X_batch, X_support, W, QPU_SIZE, ZNE_NOISE_LEVELS)
        print(f"  Compile time: {ct:.3f}s")
        print(f"  Quantum tensors: {len(heinsum.quantum_tensors)}")
        print(f"  Classical tensors: {len(heinsum.classical_tensors)}")
        for i, qt in enumerate(heinsum.quantum_tensors):
            print(f"  QT[{i}] shape={qt.shape}, qubits={qt.circuit.num_qubits}")
    else:
        print(f"Unknown command: {cmd}")
        print(usage)
        sys.exit(1)
