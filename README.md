# QVM - Quantum Virtual Machine

![QVM](./docs/intro.png)

QVM is a framework for virtual optimization and distributed execution of quantum cricuits. It builds on the work of 
"Constructing a virtual two-qubit gate by sampling single-qubit operations" [[1]](#1) to allow transparent use of
binary gate virtualization, both in order to mitigate noise and allow executions of large quantum circuits on small quantum devices.

This project started from a [Bachelor's thesis](https://raw.githubusercontent.com/TUM-DSE/research-work-archive/main/archive/2022/summer/docs/bsc_tornow_dqs_a_framework_for_efficient_distributed_simulation_of_large_quantum_circuits.pdf) at TU Munich.

## Installation
```shell
pip install qvm # TODO
```

## Quick Start
### Cutting Circuits

In this example, we cut a circuit into 2 fragments by specifying
the qubit-groups of the fragments. 
The circuit is cut by inserting a virtual gate (graphically
represented as a barrier).

```python
from qiskit import QuantumCircuit
from vqc.cut import cut, QubitGroups

circuit = QuantumCircuit(4)
circuit.h([0, 1, 2, 3])
circuit.cx(0, 1)
circuit.cx(2, 3)
circuit.cx(1, 2)
circuit.cx(0, 1)
circuit.cx(2, 3)
circuit.measure_all()

frag1 = {circuit.qubits[0], circuit.qubits[1]}
frag2 = {circuit.qubits[2], circuit.qubits[3]}

cut_circ = cut(circuit, QubitGroups([frag1, frag2]))
print(cut_circ)
"""
         ┌───┐              ░ ┌─┐         
frag0_0: ┤ H ├──■───────■───░─┤M├─────────
         ├───┤┌─┴─┐ ░ ┌─┴─┐ ░ └╥┘┌─┐      
frag0_1: ┤ H ├┤ X ├─░─┤ X ├─░──╫─┤M├──────
         ├───┤└───┘ ░ └───┘ ░  ║ └╥┘┌─┐   
frag1_0: ┤ H ├──■───░───■───░──╫──╫─┤M├───
         ├───┤┌─┴─┐ ░ ┌─┴─┐ ░  ║  ║ └╥┘┌─┐
frag1_1: ┤ H ├┤ X ├───┤ X ├─░──╫──╫──╫─┤M├
         └───┘└───┘   └───┘ ░  ║  ║  ║ └╥┘
 meas: 4/══════════════════════╩══╩══╩══╩═
                               0  1  2  3 
"""
print(cut_circ.fragments)
for frag in cut_circ.fragments:
    print(cut_circ.fragment_as_circuit(frag))
"""
[Fragment(2, 'frag0'), Fragment(2, 'frag1')]
         ┌───┐              ░ ┌─┐   
frag0_0: ┤ H ├──■───────■───░─┤M├───
         ├───┤┌─┴─┐ ░ ┌─┴─┐ ░ └╥┘┌─┐
frag0_1: ┤ H ├┤ X ├─░─┤ X ├─░──╫─┤M├
         └───┘└───┘ ░ └───┘ ░  ║ └╥┘
 meas: 4/══════════════════════╩══╩═
                               0  1 
         ┌───┐┌───┐   ┌───┐ ░    ┌─┐
frag1_0: ┤ H ├┤ X ├───┤ X ├─░────┤M├
         ├───┤└─┬─┘ ░ └─┬─┘ ░ ┌─┐└╥┘
frag1_1: ┤ H ├──■───░───■───░─┤M├─╫─
         └───┘      ░       ░ └╥┘ ║ 
 meas: 4/══════════════════════╩══╩═
                               2  3 
"""
```

### Executing

### Cut Transpilers


You can use transpiler passes to automatically virtualize binary gates in a given circuits.
In this example we use the `qvm.cut.Bisection` transpiler pass,
which replaces binary gates with virtual gates to distribute 
the circuit into two equally sized fragments.

```python
from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
from qiskit.transpiler import PassManager

from qvm.cut import Bisection
from qvm.circuit import DistributedCircuit
from qvm.executor.executor import execute

# initialize a 4-qubit circuit
circuit = QuantumCircuit.from_qasm_file("examples/qasm/hamiltonian.qasm")

# build and run a transpiler using the bisection pass.
pass_manager = PassManager(Bisection())
cut_circ = pass_manager.run(circuit)

dist_circ = DistributedCircuit.from_circuit(cut_circ)
print(dist_circ)

result = execute(dist_circ, AerSimulator(), 1000)
print(result)
``` -->


## References

<a id="1">[1]</a> 
Mitarai, Kosuke, and Keisuke Fujii. "Constructing a virtual two-qubit gate by sampling single-qubit operations." New Journal of Physics 23.2 (2021): 023021.

