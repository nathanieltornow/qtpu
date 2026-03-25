# qTPU: Hybrid Tensor Networks for Quantum-Classical Acceleration

qTPU is a compiler and runtime for hybrid quantum-classical computation using
hybrid tensor networks (hTNs). It provides a declarative `HEinsum` interface
that fuses quantum circuit tensors (qTensors) and classical tensors (cTensors)
into a single contraction, then executes across QPUs and GPUs.

## Installation

```bash
# Recommended
uv pip install git+https://github.com/nathanieltornow/qtpu

# Or with pip
pip install git+https://github.com/nathanieltornow/qtpu
```

## Quick Start

### 1. Circuit Cutting

Cut a large circuit into smaller subcircuits and execute as a hybrid tensor network:

```python
import qtpu
from qtpu import HEinsumRuntime
from evaluation.benchmarks import get_benchmark

circuit = get_benchmark("dist-vqe", circuit_size=20, cluster_size=10)
cut_circuit = qtpu.cut(circuit, max_size=10)
heinsum = qtpu.circuit_to_heinsum(cut_circuit)

runtime = HEinsumRuntime(heinsum, backend="cudaq")
runtime.prepare()
result, timing = runtime.execute()
```

### 2. Hybrid ML Model with ISwitch

Build a quantum kernel with classical weights using `ISwitch` and `HEinsum`:

```python
from qiskit.circuit import QuantumCircuit, Parameter
from qtpu import ISwitch, QuantumTensor, CTensor, HEinsum, HEinsumRuntime
import numpy as np, torch

batch_param = Parameter("batch")
iswitch = ISwitch(batch_param, num_qubits=2, size=4,
                   selector=lambda i: make_feature_circuit(i))

qc = QuantumCircuit(4)
qc.append(iswitch, [0, 1])
qc.measure_all()

qtensor = QuantumTensor(qc)
weights = CTensor(np.random.randn(4), inds=("batch",))
heinsum = HEinsum(qtensors=[qtensor], ctensors=[weights],
                  input_tensors=[], output_inds=())

runtime = HEinsumRuntime(heinsum, backend="cudaq")
runtime.prepare()
result, timing = runtime.execute()
```

### 3. Error Mitigation with ISwitch

Represent PEC basis operations as a single qTensor -- qTPU avoids
enumerating all 4^N circuit variants:

```python
from qiskit.circuit import QuantumCircuit, Parameter
from qtpu import ISwitch, QuantumTensor, CTensor, HEinsum
import numpy as np

pec_param = Parameter("pec_0")
iswitch = ISwitch(pec_param, num_qubits=1, size=4,
                   selector=lambda i: pauli_basis_circuit(i))

qc = QuantumCircuit(2)
qc.append(iswitch, [0])
qc.h(1)
qc.cx(0, 1)
qc.measure_all()

qtensor = QuantumTensor(qc)
coeffs = CTensor(np.array([1.03, -0.01, -0.01, -0.01]), inds=("pec_0",))
heinsum = HEinsum(qtensors=[qtensor], ctensors=[coeffs],
                  input_tensors=[], output_inds=())
```

## Examples

See [`examples/`](./examples/) for complete, runnable scripts:

- **`cutting.py`** -- Distributed VQE with circuit cutting
- **`hybrid_ml.py`** -- Hybrid quantum-classical ML model
- **`error_mitigation.py`** -- PEC error mitigation via hTNs

## Citation

```bibtex
@inproceedings{tornow2026qtpu,
  title     = {{qTPU}: Hybrid Tensor Networks for Quantum-Classical Acceleration},
  author    = {Tornow, Nathaniel and Mendl, Christian B. and Bhatotia, Pramod},
  booktitle = {Proceedings of the 20th USENIX Symposium on Operating Systems
               Design and Implementation (OSDI '26)},
  year      = {2026},
}
```
