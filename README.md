# qTPU: Hybrid Tensor Networks for Quantum-Classical Acceleration

qTPU is an end-to-end system for unified, flexible, and scalable hybrid quantum-classical computing.
It introduces the **hybrid tensor network (hTN)** abstraction, a programming model, compiler, and runtime that bridge quantum and classical accelerators under a single representation.

> **Paper:** *qTPU: Hybrid Tensor Networks for Quantum-Classical Acceleration*, OSDI '26.

## Getting Started

### Prerequisites

- Python 3.11 -- 3.13
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

```bash
# Clone the repository
git clone https://github.com/nathanieltornow/qtpu.git
cd qtpu

# Install with uv
uv sync
```

### Hello World: Circuit Cutting

This example cuts a 20-qubit circuit into smaller subcircuits and converts it to an hTN:

```python
from qiskit.circuit.random import random_circuit
import qtpu

# Build a 20-qubit random circuit
qc = random_circuit(20, depth=5, seed=42)

# Cut into subcircuits of at most 15 qubits, optimizing the cost tradeoff
cut_circ = qtpu.cut(qc, max_size=15)

# Convert to a hybrid tensor network (hTN)
htn = qtpu.circuit_to_heinsum(cut_circ)
print(f"Quantum tensors: {len(htn.quantum_tensors)}")
print(f"Classical tensors: {len(htn.classical_tensors)}")
```

See [`examples/`](./examples/) for more examples including hybrid ML, error mitigation, and batch training.

### Docker

A GPU-enabled Docker image is available:

```bash
docker build -t qtpu .
docker run --gpus all -it qtpu
```

## Detailed Instructions: Reproducing Paper Results

All evaluation results from the paper can be reproduced from the scripts in `evaluation/` and the pre-collected logs in `logs/`.

### Repository Structure

```
src/qtpu/           # Core library: compiler, runtime, transforms
evaluation/         # All benchmark scripts (run + plot)
  compiler/         # §8.4  Compiler tradeoff and scalability (Fig. 9, 10, Table 1)
  runtime/          # §8.5  Runtime analysis and multi-QPU scaling (Fig. 12)
  hardware/         # §8.4  Hardware validation on IBM Marrakesh (Fig. 11)
  use_cases/
    scale/          # §8.7  Scalable hybrid computing vs QAC (Fig. 13)
    hybrid_ml/      # §8.6  Hybrid ML: batch vs qTPU (Fig. 14)
    error_mitigation/ # §8.8  QEM vs Mitiq (Fig. 15)
    end_to_end/     # §8.9  End-to-end composability (Fig. 16)
logs/               # Pre-collected benchmark results (JSONL)
examples/           # Standalone usage examples
```

### Reproducing Figures from Logs

Each evaluation subsection has a `plot.py` that reads from `logs/` and generates the corresponding paper figure. No hardware or long computation is needed:

```bash
# §8.4 Compiler Pareto frontiers (Fig. 9) and scalability (Fig. 10)
uv run python -m evaluation.compiler.plot

# §8.4 Hardware fidelity: Marrakesh + noise sim + Pareto (Fig. 11)
uv run python -m evaluation.hardware.plot_qnn

# §8.5 Runtime analysis (Fig. 12)
uv run python -m evaluation.runtime.plot

# §8.7 Scalable hybrid computing (Fig. 13)
uv run python -m evaluation.use_cases.scale.plot

# §8.6 Hybrid ML (Fig. 14)
uv run python -m evaluation.use_cases.hybrid_ml.plot

# §8.8 Quantum error mitigation (Fig. 15)
uv run python -m evaluation.use_cases.error_mitigation.plot

# §8.9 End-to-end composability (Fig. 16)
uv run python -m evaluation.use_cases.end_to_end.plot
```

Generated figures are saved to `plots/`.

### Re-running Benchmarks

Each evaluation subsection also has a `run.py` that regenerates the logs from scratch. These are computationally expensive (minutes to hours depending on the benchmark):

```bash
# Example: re-run the compiler benchmark
uv run python -m evaluation.compiler.run

# Example: re-run the end-to-end composability benchmark
uv run python -m evaluation.use_cases.end_to_end.run
```

**Hardware benchmarks** (`evaluation/hardware/`) require access to IBM Quantum hardware (IBM Marrakesh). The noise-model simulations can be re-run locally:

```bash
# Pauli-noise simulation sweep (Fig. 11b)
uv run python -m evaluation.hardware.sim_sweep_qnn

# Pareto frontier under noise model (Fig. 11c)
uv run python -m evaluation.hardware.sim_pareto_80q
```

### Log Format

All benchmark results are stored as JSONL files in `logs/`. Each line is a self-contained JSON object with the benchmark parameters and results, enabling independent analysis.

## License

See [LICENSE](./LICENSE) for details.
