# VQC - Virtual Quantum Computing

![QVM](./docs/img/intro.png)

VQC is a framework for virtual optimization and distributed execution of quantum cricuits. It builds on the work of 
"Constructing a virtual two-qubit gate by sampling single-qubit operations" [[1]](#1) to allow transparent use of
binary gate virtualization, both in order to mitigate noise and allow executions of large quantum circuits on small quantum devices.

This project started from a [Bachelor's thesis](https://raw.githubusercontent.com/TUM-DSE/research-work-archive/main/archive/2022/summer/docs/bsc_tornow_dqs_a_framework_for_efficient_distributed_simulation_of_large_quantum_circuits.pdf) at TU Munich.

## Installation
```shell
pip install vqc
```

## Getting Started

```python
import vqc
from qiskit import QuantumCircuit

circuit = QuantumCircuit(...)
circuit.h(0)
# ...

virt_circuit = vqc.cut(circuit, num_fragments=3)

sampled_circs = vqc.sample(virt_circuit)
# -> list[dict[id, circuit]]

sample_results = custom_execute_sample_circuits()
# -> Dict[id, Counts]

knitted_result = vqc.knit(sample_results)
# probdistr


```

See [a short tutorial](./quickstart.ipynb).

## References

<a id="1">[1]</a> 
Mitarai, Kosuke, and Keisuke Fujii. "Constructing a virtual two-qubit gate by sampling single-qubit operations." New Journal of Physics 23.2 (2021): 023021.

