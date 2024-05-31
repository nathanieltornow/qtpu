# QVM - Quantum Virtual Machine

QVM is a framework for the scalable execution of large quantum circuits.

For this, QVM compiles circuits into **hybrid tensor networks**, which consist of quantum- and classical-tensor, where quantum tensors
consist of smaller circuits suitable for QPUs.
To contraction of a hybrid tensor network yields the result of the original circuit and can be run in a co-processing routine on quantum- and classical accelerators.


## Quickstart

```shell
pip install git+https://github.com/nathanieltornow/qvm
```

### Basic Example

```python
from qiskit import QuantumCircuit
import qvm


circuit = QuantumCircuit(20)
circuit.h(0)
# ...

# cut the circuit into a hybrid tensor network with smaller quantum-tensors
hybrid_tn = qvm.cut(circuit, oracle=qvm.compiler.NumQubitsOracle(N//2))

# contract the hybrid tensor network running on both quantum and classical devices
result = qvm.contract(hybrid_tn, shots=100000)
```

See [./docs](./docs/) for more examples and explanations.

## GPU 

### Using Docker

Build the image with all its dependencies:
```sh
docker build -t qvm .
```

Run the docker container while mounting the project's directory into `/home/cuquantum/qvm`:
```sh
docker run --gpus all -it --rm qvm
```