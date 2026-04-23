# qTPU: Hybrid Tensor Networks for Quantum-Classical Acceleration

> **Paper:** [qTPU: Hybrid Tensor Networks for Quantum-Classical Acceleration](https://arxiv.org/abs/2410.15080), OSDI '26.

qTPU is an end-to-end system for unified hybrid quantum-classical computing. It introduces the **hybrid tensor network (hTN)** abstraction — a programming model, compiler, and runtime that bridge quantum and classical accelerators under a single representation.

## Installation

Requires Python 3.11–3.13. We recommend [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/nathanieltornow/qtpu.git && cd qtpu
uv sync
```

Or with pip:

```bash
pip install git+https://github.com/nathanieltornow/qtpu.git
```

## Quick Example

```python
import qtpu
from qiskit.circuit.random import random_circuit

# Build a 20-qubit circuit
qc = random_circuit(20, depth=5, seed=42)

# Compile: partition into ≤18-qubit subcircuits
htn = qtpu.compile_to_heinsum(qc, max_size=18)

print(f"Quantum tensors:  {len(htn.quantum_tensors)}")
print(f"Classical tensors: {len(htn.classical_tensors)}")
```

See [`examples/`](./examples/) for circuit cutting, hybrid ML, and error mitigation.

## Reproducing Paper Results

All figures from the paper can be reproduced from the pre-collected logs in `logs/`. See **[REPRODUCE.md](./REPRODUCE.md)** for step-by-step instructions.

## Citation

```bibtex
@inproceedings{tornow2026qtpu,
  title     = {{qTPU}: Hybrid Tensor Networks for Quantum-Classical Acceleration},
  author    = {Tornow, Nathaniel and Mendl, Christian B. and Bhatotia, Pramod},
  booktitle = {OSDI},
  year      = {2026}
}
```
