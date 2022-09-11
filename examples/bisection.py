from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
from qiskit.transpiler import PassManager

from qvm.cut import Bisection
from qvm.circuit import DistributedCircuit
from qvm.executor.executor import execute


# initialize a 4-qubit circuit
circuit = QuantumCircuit.from_qasm_file("examples/qasm/circuit1.qasm")

# build and run a transpiler using the bisection pass.
pass_manager = PassManager(Bisection())
cut_circ = pass_manager.run(circuit)

dist_circ = DistributedCircuit.from_circuit(cut_circ)
print(dist_circ)

result = execute(dist_circ, AerSimulator(), 1000)
print(result)

from qvm.bench.fidelity import fidelity

fid = fidelity(circuit, result)
print(fid)
