# qTPU - Quantum Tensor Processing Unit

`qTPU` is a framework for the scalable execution of large quantum circuits.

For this, qTPU compiles circuits into **hybrid tensor networks**, consisting of quantum- and classical-tensors, where quantum tensors are represented by smaller quantum circuits.
For this, qTPU makes use of **quasiprobability decompostiton (QPD)**.

To contraction of a hybrid tensor network returns the result of the original circuit and can be run in a co-processing routine on quantum- and classical accelerators.


## Quickstart

```shell
pip install git+https://github.com/nathanieltornow/qtpu
```

### Basic Example

```python
from qiskit import QuantumCircuit
import qtpu

N = 20
circuit = QuantumCircuit(N)
circuit.h(0)
# ...
circuit.measure_all()

# cut the circuit into a hybrid tensor network with quantum-tensors
# half the size of the original cricuit
hybrid_tn = qtpu.cut(circuit, oracle=qtpu.compiler.NumQubitsOracle(N//2))

# contract the hybrid tensor network running on both quantum and classical devices
result = qtpu.contract(hybrid_tn, shots=100000)
```

See [./docs](./docs/) for more examples and explanations.

## GPU 

### Using Docker

Build the image with all its dependencies:
```sh
docker build -t qtpu .
```

Run the docker container while mounting the project's directory
```sh
docker run --gpus all -it --rm -v $(pwd):/home/cuquantum/qtpu qtpu
```

To contract using `cuTensorNet`, use 
```python
from qtpu.cutensor import contract_gpu

result = contract_gpu(hybrid_tn, shots=10000)
```

Note that this contraction does only support the computation of expectation values.