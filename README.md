# qTPU - Hybrid Quantum-Classical Processing using Tensor Networks

## Quickstart

```shell
pip install git+https://github.com/nathanieltornow/qtpu
```

### Basic Example

```python
from qiskit import QuantumCircuit
from qiskit_aer.primitives import EstimatorV2
import qtpu

N = 20
circuit = QuantumCircuit(N)
# ...

# cut the circuit into subcircuits a quater the size of the original cricuit
cut_circuit = qtpu.cut(circuit, num_qubits=N//4, show_progress_bar=True, n_trials=10)

# convert the circuit into a hybrid tensor network (h-TN)
hybrid_tn = qtpu.circuit_to_hybrid_tn(cut_circuit, num_samples=np.inf)

# contract the hybrid tensor network to get the <ZZ..Z>
est = EstimatorV2()
res = qtpu.contract(htn, est)

```

See [./docs](./docs/) for more examples and explanations.

