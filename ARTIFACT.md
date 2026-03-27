# Artifact Evaluation — qTPU (OSDI '26)

## Overview

This artifact reproduces all evaluation results from the paper:
**"qTPU: Hybrid Tensor Networks for Quantum-Classical Acceleration"**

| Figure | Description | Script | Est. Time |
|--------|------------|--------|-----------|
| Fig 9 | Compiler tradeoff (Pareto frontiers) | `evaluation/compiler/` | ~30 min |
| Fig 10 | Compiler scalability (error vs size) | `evaluation/compiler/` | (same run) |
| Table 1 | Compile time scaling (VQE-SU2) | `evaluation/compiler/` | (same run) |
| Fig 11 | Runtime analysis (scaling, multi-QPU, vs cuTensorNet) | `evaluation/runtime/` | ~40 min |
| Fig 12 | Circuit knitting scalability (QNN) | `evaluation/use_cases/scale/` | ~30 min |
| Fig 13 | Hybrid ML (compile time + code size) | `evaluation/use_cases/hybrid_ml/` | ~20 min |
| Fig 14 | Error mitigation (QTPU vs Mitiq) | `evaluation/use_cases/error_mitigation/` | ~3 min |

**Total estimated time: ~2 hours** (single-threaded, CPU only).

## Requirements

- **No quantum hardware (QPU) required.** All quantum execution times are estimated
  via Qiskit transpilation to IBM FakeMarrakesh with ASAP scheduling.
- **No GPU required** for the primary results. GPU (NVIDIA + cuTensorNet) is only
  needed for the classical-simulation baseline in Fig 11(c).
- **Runs on macOS and Linux.** No Linux requirement — tested on both platforms.
- Python 3.10+, ~8 GB RAM, ~10 GB disk (for dependencies).

## Setup

### Option A: Docker (recommended)

```bash
docker build -t qtpu .
docker run --rm -v $(pwd)/output:/app/output qtpu bash -c '
  python -m evaluation.run_all && cp -r plots logs /app/output/
'
```

### Option B: Local installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/nathanieltornow/qtpu.git
cd qtpu
git checkout bench-osdi
uv sync

# Verify installation
uv run python -c "import qtpu; print('OK')"
```

## Reproducing Results

### Full evaluation (~2 hours)

```bash
uv run python -m evaluation.run_all
```

This runs all benchmarks and generates all plots. Output:
- `logs/` — Raw JSONL benchmark data
- `plots/` — PDF figures matching the paper

### Quick smoke test (~3 minutes)

```bash
uv run python -m evaluation.run_all --quick
```

Runs only Fig 14 (error mitigation) as a fast sanity check.

### Single figure

```bash
uv run python -m evaluation.run_all --fig 14    # Error mitigation (~3 min)
uv run python -m evaluation.run_all --fig 13    # Hybrid ML (~20 min)
uv run python -m evaluation.run_all --fig 12    # Circuit knitting (~30 min)
uv run python -m evaluation.run_all --fig 9-10  # Compiler (~30 min)
uv run python -m evaluation.run_all --fig 11    # Runtime (~40 min)
```

### Regenerate plots from existing logs

```bash
uv run python -m evaluation.run_all --plots-only
```

## Output Files

After a full run, the following files are produced:

```
plots/
  pareto_frontiers.pdf      # Fig 9  — Compiler Pareto frontiers
  scalability.pdf           # Fig 10 — Compiler scalability
  compile_times.pdf         # Table 1 — Compile time scaling
  runtime_analysis.pdf      # Fig 11 — Runtime analysis
  scale_qnn.pdf             # Fig 12 — Circuit knitting scalability
  hybrid_ml/benchmark.pdf   # Fig 13 — Hybrid ML
  error_mitigation.pdf      # Fig 14 — Error mitigation

logs/
  compiler/                 # Raw data for Figs 9, 10, Table 1
  runtime/                  # Raw data for Fig 11
  scale/                    # Raw data for Fig 12
  hybrid_ml/                # Raw data for Fig 13
  error_mitigation/         # Raw data for Fig 14
```

## What the Benchmarks Measure

All benchmarks are **purely classical computations** that measure host-side overhead:

| Metric | How measured | QPU? |
|--------|-------------|------|
| Compilation time | `time.perf_counter()` around `qtpu.cut()` and optimizer | No |
| Classical cost (FLOPs) | Contraction path analysis via cotengra | No |
| Quantum error | Gate error model (10^-3 1Q, 10^-2 2Q) | No |
| QPU time (estimated) | Qiskit transpile to FakeMarrakesh + ASAP scheduling | No |
| Code size (LoC) | Count lines of generated CUDA-Q kernel code | No |
| Circuit count | Count expanded circuits from qTensor shapes | No |

The key insight is that qTPU's benefits (reduced compilation time, smaller code,
fewer circuits, lower classical overhead) are all measurable without executing
any quantum circuits. The paper's claims about quantum error are based on an
analytical error model, not empirical QPU measurements.

## Baselines

| Baseline | Source | Version |
|----------|--------|---------|
| QAC (Qiskit Addon Cutting) | `qiskit-addon-cutting` | >=0.10.0 |
| cuTensorNet | `cutensornet-cu12` | >=2.11.0 |
| Mitiq (reimplemented) | `evaluation/use_cases/error_mitigation/run.py` | N/A |
| BATCH method | `qtpu.runtime.baseline.run_batch()` | N/A |

Mitiq's circuit generation is reimplemented directly (matching its PEC/twirling/ZNE
approach) to avoid dependency issues, since Mitiq's API has evolved. The
reimplementation generates circuits identically to Mitiq's internal logic.

## Expected Variability

- **Compilation times** may vary by ~2x depending on CPU speed, but relative
  speedups (qTPU vs baselines) should be consistent.
- **Classical cost** (FLOPs) and **code size** (LoC) are deterministic.
- **Quantum error** estimates are deterministic (analytical model).
- **Pareto frontier** points may vary slightly due to randomized hyperparameter
  optimization (controlled by seed where possible).
