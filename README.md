# QVM - Quantum Virtual Machine

QVM is a framework for virtual optimization and distributed execution of quantum cricuits. It builds on the work of 
"Constructing a virtual two-qubit gate by sampling single-qubit operations" [[1]](#1) to allow transparent use of
binary gate virtualization, both in order to mitigate noise and allow executions of large quantum circuits on small quantum devices.

## Installation
```shell
pip install qvm # TODO
```

## Examples
Simple example of virtualizing every binary gate between two qubits.

```python
from qiskit import QuantumCircuit
from qvm.cut import cut_qubit_connection
from qvm.executor import execute

circuit = QuantumCircuit(2, 2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()


```



## References

<a id="1">[1]</a> 
Mitarai, Kosuke, and Keisuke Fujii. "Constructing a virtual two-qubit gate by sampling single-qubit operations." New Journal of Physics 23.2 (2021): 023021.

