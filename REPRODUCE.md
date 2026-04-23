# Reproducing Paper Results

This document provides step-by-step instructions to reproduce every figure and table from the paper using the pre-collected logs in `logs/`.

## Prerequisites

```bash
git clone https://github.com/nathanieltornow/qtpu.git && cd qtpu
uv sync
```

## Repository Structure

```
src/qtpu/           Core library (compiler, runtime, transforms)
evaluation/         Benchmark scripts (run + plot)
  compiler/         §8.4  Compiler tradeoff and scalability
  runtime/          §8.5  Runtime analysis and multi-QPU scaling
  hardware/         §8.4  Hardware validation (IBM Marrakesh)
  use_cases/
    scale/          §8.7  Scalable hybrid computing vs QAC
    hybrid_ml/      §8.6  Hybrid ML: batch vs qTPU
    error_mitigation/ §8.8  QEM vs Mitiq
    end_to_end/     §8.9  End-to-end composability
logs/               Pre-collected benchmark results (JSONL)
```

## Reproducing Figures from Logs

Each figure can be generated in seconds from the pre-collected logs. Generated plots are saved to `plots/`.

### §8.4 Compiler (Fig. 9, Fig. 10, Table 1)

```bash
# Fig. 9: Pareto frontiers — qTPU vs QAC
# Fig. 10: Compiler optimality analysis across circuit sizes
# Table 1: Compile time scaling for VQE-SU2
uv run python -m evaluation.compiler.plot
```

**Expected output:** `plots/plot_pareto_frontiers/` and `plots/plot_solutions_by_size/`

### §8.4 Hardware Validation (Fig. 11)

```bash
# Fig. 11a: IBM Marrakesh fidelity (10–80 qubits)
# Fig. 11b: Pauli-noise simulation (20–150 qubits)
# Fig. 11c: Pareto frontier traversal at 80 qubits
uv run python -m evaluation.hardware.plot_qnn
```

**Expected output:** `plots/plot_qnn/`

### §8.5 Runtime Analysis (Fig. 12)

```bash
# Fig. 12a: Runtime scaling with circuit size
# Fig. 12b: Multi-QPU speedup (1–16 QPUs)
# Fig. 12c: qTPU vs cuTensorNet on 100q Dist-VQE
uv run python -m evaluation.runtime.plot
```

**Expected output:** `plots/plot_runtime_analysis/`

### §8.6 Hybrid Machine Learning (Fig. 14)

```bash
# Fig. 14: Compilation time and code size — batch vs qTPU
uv run python -m evaluation.use_cases.hybrid_ml.plot
```

**Expected output:** `plots/plot_hybrid_ml_benchmark/`

### §8.7 Scalable Hybrid Computing (Fig. 13)

```bash
# Fig. 13: QNN on 10-qubit QPU — runtime, circuits, FLOPs
uv run python -m evaluation.use_cases.scale.plot
```

**Expected output:** `plots/plot_scale_comparison/`

### §8.8 Quantum Error Mitigation (Fig. 15)

```bash
# Fig. 15: QEM compile time and code size — qTPU vs Mitiq
uv run python -m evaluation.use_cases.error_mitigation.plot
```

**Expected output:** `plots/plot_error_mitigation_comparison/`

### §8.9 End-to-End Composability (Fig. 16)

```bash
# Fig. 16: End-to-end time, circuits, and code — qTPU vs baseline
uv run python -m evaluation.use_cases.end_to_end.plot
```

**Expected output:** `plots/plot_end_to_end/`

## Re-running Benchmarks from Scratch

Each evaluation also has a `run.py` that regenerates logs from scratch. These are computationally expensive (minutes to hours):

```bash
# Compiler benchmarks (§8.4) — ~10 min
uv run python -m evaluation.compiler.run

# Runtime analysis (§8.5) — ~30 min
uv run python -m evaluation.runtime.run

# Scalable hybrid computing (§8.7) — ~5 min
uv run python -m evaluation.use_cases.scale.run

# Hybrid ML (§8.6) — ~10 min
uv run python -m evaluation.use_cases.hybrid_ml.run

# Error mitigation (§8.8) — ~5 min
uv run python -m evaluation.use_cases.error_mitigation.run

# End-to-end composability (§8.9) — ~30 min
uv run python -m evaluation.use_cases.end_to_end.run
```

### Hardware Benchmarks (§8.4, Fig. 11)

**Real hardware (Fig. 11a)** requires access to IBM Quantum (IBM Marrakesh):

```bash
export IBM_TOKEN=<your-ibm-quantum-token>
uv run python -m evaluation.hardware.sweep_marrakesh_qnn
```

**Noise-model simulations (Fig. 11b, 11c)** can be run locally:

```bash
# Pauli-noise sweep to 150 qubits (~20 min)
uv run python -m evaluation.hardware.sim_sweep_qnn

# Pareto frontier at 80 qubits (~60 min)
uv run python -m evaluation.hardware.sim_pareto_80q
```

## Log Format

All results are stored as JSONL files in `logs/`. Each line is a self-contained JSON object with benchmark parameters and results, enabling independent analysis.
