import logging
from typing import List
from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
from qiskit.transpiler import PassManager

from qvm.cut import Bisection, LadderDecomposition
from qvm.circuit import VirtualCircuit
from qvm.execution.executor import execute

logging.basicConfig(level=logging.INFO)

circuit = QuantumCircuit.from_qasm_file("examples/qasm/hamiltonian.qasm")

pass_manager = PassManager(LadderDecomposition(3))
cut_circ = pass_manager.run(circuit)

print(cut_circ)

vcirc = VirtualCircuit.from_circuit(cut_circ)
print(vcirc)


result = execute(vcirc, AerSimulator(), 1000)
print(result)

from qvm.bench.fidelity import fidelity

print(fidelity(circuit, result))
