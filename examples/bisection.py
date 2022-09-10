import logging
from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
from qiskit.transpiler import PassManager

from qvm.cut import Bisection
from qvm.circuit import DistributedCircuit
from qvm.executor.executor import execute

logging.basicConfig(level=logging.INFO)

circuit = QuantumCircuit.from_qasm_file("examples/qasm/hamiltonian.qasm")

pass_manager = PassManager(Bisection())
cut_circ = pass_manager.run(circuit)

print(cut_circ)

vcirc = DistributedCircuit.from_circuit(cut_circ)
print(vcirc)


result = execute(vcirc, AerSimulator(), 1000)
print(result)

from qvm.bench.fidelity import fidelity

print(fidelity(circuit, result))
