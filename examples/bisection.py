from typing import List
from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
from qiskit.transpiler import PassManager

from qvm.cut import Bisection
from qvm.circuit import VirtualCircuit
from qvm.execution.executor import execute

circuit = QuantumCircuit.from_qasm_file("examples/qasm/hamiltonian.qasm")
circuit.barrier(circuit.qregs[0])
circuit.measure_all()

pass_manager = PassManager(Bisection())
cut_circ = pass_manager.run(circuit)

print(cut_circ)

vcirc = VirtualCircuit.from_circuit(cut_circ)
print(vcirc)


result = execute(vcirc, AerSimulator(), 10000)
print(result)

from qvm.bench.fidelity import fidelity

print(fidelity(circuit, result))