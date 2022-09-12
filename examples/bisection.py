from typing import List
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import PassManager
from qiskit.providers.aer import AerSimulator

from vqc.converters import circuit_to_connectivity_graph

from vqc.cut import Bisection
from vqc.circuit import DistributedCircuit
from vqc.executor.executor import execute
from vqc.device import Device
from vqc.prob import ProbDistribution


class AerDevice(Device):
    def run(self, circuits: List[QuantumCircuit], shots: int) -> List[ProbDistribution]:
        backend = AerSimulator()
        t_circs = transpile(circuits, backend, optimization_level=3)
        if len(t_circs) == 1:
            return [
                ProbDistribution.from_counts(
                    backend.run(t_circs[0], shots=shots).result().get_counts()
                )
            ]
        cnts = backend.run(t_circs, shots=shots).result().get_counts()
        return [ProbDistribution.from_counts(c) for c in cnts]


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

result = execute(dist_circ, AerDevice(), 1000)
print(result)

from vqc.bench.fidelity import fidelity

fid = fidelity(circuit, result)
print(fid)
