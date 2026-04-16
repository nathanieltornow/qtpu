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
            elif op.num_qubits == 1 and op.name not in (
                "measure",
                "barrier",
                "reset",
                "qpd_measure",
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
    num_zne_samples: int,
) -> dict:
    """Run baseline pipeline: QAC cutting + Mitiq-style ZNE + manual batching.

    Returns dict with timing and circuit metrics.
    """
    from qiskit_addon_cutting import (
        cut_wires,
        expand_observables,
        generate_cutting_experiments,
        partition_problem,
    )
    from qiskit_addon_cutting.automated_cut_finding import (
        DeviceConstraints,
        OptimizationParameters,
        find_cuts,
    )
    from qiskit.quantum_info import SparsePauliOp

    n_batch = len(X_batch)
    n_support = len(X_support)
    total_circuits = 0
    total_compile_time = 0.0
    total_quantum_time = 0.0
    total_code_lines = 0

    for b in range(n_batch):
        for s in range(n_support):
            # Build concrete circuit for this (batch, support) pair
            qc = QuantumCircuit(num_qubits)
            qc.compose(
                create_feature_map(num_qubits, X_batch[b], NUM_LAYERS), inplace=True
            )
            qc.compose(
                create_feature_map(num_qubits, X_support[s], NUM_LAYERS).inverse(),
                inplace=True,
            )

            # --- QAC cutting ---
            start = perf_counter()
            try:
                cut_circuit, _ = find_cuts(
                    qc,
                    OptimizationParameters(),
                    DeviceConstraints(qubits_per_subcircuit=max_subcircuit_size),
                )
                qc_w_ancilla = cut_wires(cut_circuit)
                observable = SparsePauliOp(["Z" * num_qubits])
                observables_expanded = expand_observables(
                    observable.paulis, qc, qc_w_ancilla
                )
                partitioned = partition_problem(
                    circuit=qc_w_ancilla, observables=observables_expanded
                )
                subcircuits = partitioned.subcircuits
                subobservables = partitioned.subobservables
                subexperiments, _ = generate_cutting_experiments(
                    circuits=subcircuits,
                    observables=subobservables,
                    num_samples=np.inf,
                )
            except Exception:
                # If cutting fails, use the original circuit
                subcircuits = {"A": qc}
                subexperiments = {"A": [qc]}

            cut_time = perf_counter() - start
            total_compile_time += cut_time

            # Collect all subcircuit experiments
            all_subcirc_experiments = [
                exp for exps in subexperiments.values() for exp in exps
            ]

            # --- Mitiq-style ZNE: enumerate circuits at each noise level ---
            zne_circuits = []
            rng = np.random.default_rng(42)
            samples_per_level = max(1, num_zne_samples // len(noise_levels))

            for level in noise_levels:
                for exp_circuit in all_subcirc_experiments:
                    for _ in range(samples_per_level):
                        folded = exp_circuit.copy()
                        if level > 1:
                            # Apply noise folding: append gate^-1 gate pairs
                            extra = QuantumCircuit(*folded.qregs, *folded.cregs)
                            for instr in exp_circuit.data:
                                if (
                                    instr.operation.num_qubits == 1
                                    and instr.operation.name
                                    not in ("measure", "barrier", "reset", "qpd_measure")
                                ):
                                    qubit_indices = [
                                        exp_circuit.find_bit(q).index
                                        for q in instr.qubits
                                    ]
                                    for _ in range((level - 1) // 2):
                                        extra.append(
                                            instr.operation.inverse(), qubit_indices
                                        )
                                        extra.append(
                                            instr.operation, qubit_indices
                                        )
                            folded = folded.compose(extra)
                        zne_circuits.append(folded)

            total_circuits += len(zne_circuits)

            # Estimate code lines (each circuit → separate kernel)
            if zne_circuits:
                avg_gates = sum(
                    len(
                        [
                            i
                            for i in circ.data
                            if i.operation.name not in ("barrier", "measure")
                        ]
                    )
                    for circ in zne_circuits
                ) / len(zne_circuits)
                lines_per_kernel = int(50 + avg_gates * 2)
                total_code_lines += len(zne_circuits) * lines_per_kernel

            # Estimate QPU time
            if zne_circuits:
                total_quantum_time += estimate_runtime(zne_circuits[:10]) * (
                    len(zne_circuits) / min(10, len(zne_circuits))
                )

    return {
        "compile_time": total_compile_time,
        "quantum_time": total_quantum_time,
        "num_circuits": total_circuits,
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
        start = perf_counter()
        heinsum, compile_time = build_e2e_qtpu(
            circuit_size, X_batch, X_support, W, QPU_SIZE, ZNE_NOISE_LEVELS
        )

        # Run with HEinsum runtime
        _, timing = run_heinsum(heinsum, skip_execution=True)

        total_time = perf_counter() - start

        # Count total code lines
        total_code_lines = 0
        for qt in heinsum.quantum_tensors:
            _, num_lines = quantum_tensor_to_cudaq(
                qt.circuit, qt.shape, kernel_name="e2e_kernel"
            )
            total_code_lines += num_lines

        return {
            "compile_time": compile_time,
            "quantum_time": timing.quantum_estimated_qpu_time,
            "classical_time": timing.classical_contraction_time,
            "total_time": total_time,
            "num_circuits": timing.num_circuits,
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
    """Benchmark baseline pipeline (QAC + Mitiq-style ZNE + manual batching)."""
    print(f"Baseline E2E: qubits={circuit_size}")

    np.random.seed(42)
    X_batch = np.random.randn(BATCH_SIZE, FEATURE_DIM) * np.pi
    X_support = np.random.randn(NUM_SUPPORT, FEATURE_DIM) * np.pi
    W = np.random.randn(NUM_SUPPORT) * 0.1

    # For baseline, use smaller batch to keep runtime manageable
    # Scale results linearly
    sample_batch = min(5, BATCH_SIZE)
    sample_support = min(5, NUM_SUPPORT)

    try:
        start = perf_counter()
        result = run_e2e_baseline(
            circuit_size,
            X_batch[:sample_batch],
            X_support[:sample_support],
            W[:sample_support],
            QPU_SIZE,
            ZNE_NOISE_LEVELS,
            num_zne_samples=len(ZNE_NOISE_LEVELS),
        )
        sample_time = perf_counter() - start

        # Scale to full batch/support size
        scale_factor = (BATCH_SIZE * NUM_SUPPORT) / (sample_batch * sample_support)

        return {
            "compile_time": result["compile_time"] * scale_factor,
            "quantum_time": result["quantum_time"] * scale_factor,
            "total_time": sample_time * scale_factor,
            "num_circuits": int(result["num_circuits"] * scale_factor),
            "total_code_lines": int(result["total_code_lines"] * scale_factor),
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
