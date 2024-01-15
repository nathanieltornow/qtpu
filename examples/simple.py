import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import TwoLocal, QuantumVolume
from qiskit_aer import AerSimulator

from qvm.runtime.runner import expval_from_counts, sample_fragments
from qvm.runtime.runners import SimRunner
from qvm.runtime.virtualizer import build_tensornetwork, build_dummy_tensornetwork
from qvm.cutter.girvan_newman import GirvanNewmanCutter
from qvm.cutter.metis import MetisCutter
from qvm.virtual_circuit import VirtualCircuit


circuit = TwoLocal(
    4,
    rotation_blocks=["rz", "ry"],
    entanglement_blocks="rzz",
    entanglement="linear",
    reps=2,
).decompose()
circuit.measure_all()


params = {param: np.random.randn() / 2 for param in circuit.parameters}

# cut_circuit = GirvanNewmanCutter(100).run(circuit)
cut_circuit = MetisCutter(2).run(circuit)

print(cut_circuit)

cut_circuit = cut_circuit.assign_parameters(params)

virtual_circuit = VirtualCircuit(cut_circuit)
print(virtual_circuit.num_instantiations())

# tn = build_dummy_tensornetwork(virtual_circuit)
# print(tn.contraction_cost(optimize="auto"))
# tn.draw(color=["frag_result", "coeff"])


results = sample_fragments(virtual_circuit, SimRunner(), shots=100000)
tn = build_tensornetwork(virtual_circuit, results)
result = tn.contract(all, optimize="auto")

circuit = circuit.assign_parameters(params)
counts = AerSimulator().run(circuit, shots=100000).result().get_counts()
print(abs(result - expval_from_counts(counts)))
