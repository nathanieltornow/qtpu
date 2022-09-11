from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qvm.converters import circuit_to_connectivity_graph

from qvm.cut import Bisection
from qvm.circuit import DistributedCircuit
from qvm.executor.executor import execute
from qvm.device import AerSimDevice


# initialize a 4-qubit circuit
# circuit = QuantumCircuit.from_qasm_file("examples/qasm/circuit1.qasm")
circuit = QuantumCircuit(2)
circuit.h(0)
circuit.h(1)
circuit.measure_all()

# build and run a transpiler using the bisection pass.
pass_manager = PassManager(Bisection())
cut_circ = pass_manager.run(circuit)

dist_circ = DistributedCircuit.from_circuit(cut_circ)
print(dist_circ)

result = execute(dist_circ, AerSimDevice(), 1000)
print(result)

from qvm.bench.fidelity import fidelity

fid = fidelity(circuit, result)
print(fid)
