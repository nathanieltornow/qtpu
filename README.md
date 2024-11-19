# qTPU - Hybrid Quantum-Classical Processing using Tensor Networks

## Quickstart

```shell
pip install git+https://github.com/nathanieltornow/qtpu
```

### Basic Example

```python
from qiskit import QuantumCircuit
import qtpu

# generate some quantumcircuit
circuit = QuantumCircuit(...)

# cut the circuit into two halves
cut_circ = qtpu.cut(circuit, num_qubits=circuit.num_qubits // 2)

# convert the circuit into a hybrid tensor network
hybrid_tn = qtpu.circuit_to_hybrid_tn(cut_circ)

for i, subcirc in enumerate(hybrid_tn.subcircuits):
    print(f"Subcircuit {i}:")
    print(subcirc)
    print("--------------------")

# evaluate the hybrid tensor network to a classical tensor network
tn = qtpu.evaluate(hybrid_tn)

# contract the classical tensor network
res = tn.contract(all, optimize="auto-hq", output_inds=[])

```

See [./examples](./examples/) for more examples and explanations.

## Paper

```text
@article{tornow2024quantum,
  title={Quantum-Classical Computing via Tensor Networks},
  author={Tornow, Nathaniel and Mendl, Christian B and Bhatotia, Pramod},
  journal={arXiv preprint arXiv:2410.15080},
  year={2024}
}
```