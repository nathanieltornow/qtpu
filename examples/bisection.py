from typing import List
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import PassManager
from qiskit.providers.aer import AerSimulator

from vqc.converters import circuit_to_connectivity_graph

from vqc.cut import Bisection
from vqc.circuit import DistributedCircuit
from vqc.executor.executor import execute
from vqc.device import Device, SimDevice
from vqc.prob import ProbDistribution

# initialize a 4-qubit circuit
circuit = QuantumCircuit.from_qasm_file("examples/qasm/circuit1.qasm")
# circuit = QuantumCircuit(2)
# circuit.h(0)
# circuit.h(1)
# circuit.measure_all()

# build and run a transpiler using the bisection pass.
pass_manager = PassManager(Bisection())
cut_circ = pass_manager.run(circuit)

dist_circ = DistributedCircuit.from_circuit(cut_circ)
print(dist_circ)

result = execute(dist_circ, 1000)
print(result)

from vqc.bench.fidelity import fidelity

fid = fidelity(circuit, result)
print(fid)
