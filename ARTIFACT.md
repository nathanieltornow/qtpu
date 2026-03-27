# Artifact Evaluation — qTPU (OSDI '26)

## Overview

This artifact reproduces all evaluation results from:
**"qTPU: Hybrid Tensor Networks for Quantum-Classical Acceleration"**

| Figure | Description | Est. Time |
|--------|------------|-----------|
| Fig 9 | Compiler tradeoff (Pareto frontiers) | ~30 min |
| Fig 10 | Compiler scalability (error vs circuit size) | (same run) |
| Table 1 | Compile time scaling (VQE-SU2) | (same run) |
| Fig 11 | Runtime analysis (scaling, multi-QPU, vs cuTensorNet) | ~40 min |
| Fig 12 | Circuit knitting scalability (QNN on 10q QPU) | ~30 min |
| Fig 13 | Hybrid ML (compile time + code size) | ~20 min |
| Fig 14 | Error mitigation (qTPU vs Mitiq) | ~3 min |

**Total: ~2 hours** on a single core.

## Requirements

- Python 3.10+, macOS or Linux
- ~8 GB RAM, ~10 GB disk
- No QPU or GPU required

All quantum execution times are *estimated* via Qiskit transpilation to
IBM FakeMarrakesh (ASAP scheduling, 1000 shots). No circuits are executed.
A GPU is only needed to reproduce the cuTensorNet baseline in Fig 11(c).

## Setup

### Docker

```bash
docker build -t qtpu .
docker run --rm -v $(pwd)/output:/app/output qtpu bash -c \
  'python -m evaluation.run_all && cp -r plots logs /app/output/'
```

### Local

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv
git clone https://github.com/nathanieltornow/qtpu.git && cd qtpu
uv sync
uv run python -c "import qtpu; print('OK')"
```

## Reproducing Results

```bash
# Full evaluation (~2 hours)
uv run python -m evaluation.run_all

# Quick smoke test (~3 minutes, Fig 14 only)
uv run python -m evaluation.run_all --quick

# Single figure
uv run python -m evaluation.run_all --fig 14     # Error mitigation
uv run python -m evaluation.run_all --fig 13     # Hybrid ML
uv run python -m evaluation.run_all --fig 12     # Circuit knitting
uv run python -m evaluation.run_all --fig 9-10   # Compiler
uv run python -m evaluation.run_all --fig 11     # Runtime

# Regenerate plots from existing logs
uv run python -m evaluation.run_all --plots-only
```

## Output

```
plots/
  pareto_frontiers.pdf      # Fig 9
  scalability.pdf           # Fig 10
  compile_times.pdf         # Table 1
  runtime_analysis.pdf      # Fig 11
  scale_qnn.pdf             # Fig 12
  hybrid_ml/benchmark.pdf   # Fig 13
  error_mitigation.pdf      # Fig 14

logs/                       # Raw JSONL data for each figure
```

## What the Benchmarks Measure

All benchmarks measure host-side overhead without executing quantum circuits:

| Metric | Method |
|--------|--------|
| Compilation time | Wall-clock around `qtpu.cut()` and optimizer |
| Classical cost | FLOPs from contraction path (cotengra) |
| Quantum error | Analytical model (10^-3 1Q, 10^-2 2Q gate error) |
| QPU time | Qiskit transpile to FakeMarrakesh + ASAP scheduling |
| Code size | Lines of generated CUDA-Q kernel code |
| Circuit count | Expanded circuits from qTensor shapes |

qTPU's benefits — faster compilation, smaller code, fewer circuits, lower
classical cost — are all measurable without running quantum hardware.

## Baselines

| Baseline | Used in | Source |
|----------|---------|--------|
| QAC (Qiskit Addon Cutting) | Figs 9–12 | `qiskit-addon-cutting` >=0.10.0 |
| cuTensorNet | Fig 11(c) | `cutensornet-cu12` >=2.11.0 (GPU only) |
| Mitiq-style | Fig 14 | Reimplemented in `evaluation/use_cases/error_mitigation/run.py` |
| BATCH | Fig 13 | `qtpu.runtime.baseline.run_batch()` |

The Mitiq baseline is reimplemented directly (PEC, Pauli twirling, ZNE) matching
Mitiq's circuit generation logic, to avoid external dependency issues.

## Expected Variability

- **Compilation times** scale with CPU speed; relative speedups stay consistent.
- **Classical cost** (FLOPs), **code size** (LoC), and **quantum error** are deterministic.
- **Pareto frontiers** may vary slightly due to randomized optimization (seeded where possible).
