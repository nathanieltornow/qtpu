from typing import List
from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator

from qvm import transpile, execute
from qvm.transpiler import VirtualizationPass, DistributedPass
from qvm.transpiler.decomposition import Bisection
from qvm.transpiler.decomposition.ladder import LadderDecomposition
from qvm.transpiler.device_map import SingleDeviceMapping
from qvm.bench.fidelity import fidelity

circuit = QuantumCircuit.from_qasm_file("examples/qasm/hamiltonian.qasm")
cp = circuit.copy()
print(circuit)

virt_passes: List[VirtualizationPass] = [LadderDecomposition(2)]
distr_passes: List[DistributedPass] = [SingleDeviceMapping(AerSimulator())]

frag_circ = transpile(circuit, virt_passes=virt_passes, distr_passes=distr_passes)

for frag in frag_circ.fragments:
    print(frag)

counts = execute(frag_circ, shots=8192)

print(fidelity(cp, counts))
