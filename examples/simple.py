import numpy as np
from qiskit.circuit.library import TwoLocal
from qiskit_aer import AerSimulator

import qvm
from qvm.compiler import girvan_newman_cut_circuit


circuit = TwoLocal(
    5,
    rotation_blocks=["rz", "ry"],
    entanglement_blocks="rzz",
    entanglement="linear",
    reps=2,
).decompose()
circuit.measure_all()


params = {param: np.random.randn() / 2 for param in circuit.parameters}

cut_circuit = girvan_newman_cut_circuit(circuit, num_fragments=2)

print(cut_circuit)


cut_circuit = cut_circuit.assign_parameters(params)

virtual_circuit = qvm.VirtualCircuit(cut_circuit)
print(virtual_circuit.num_instantiations())

result = qvm.run_virtual_circuit(virtual_circuit, shots=100000)

tn = qvm.build_dummy_tensornetwork(virtual_circuit)
print(tn.contraction_cost(optimize="auto"))


# results = sample_fragments(virtual_circuit, SimRunner(), shots=100000)
# tn = build_tensornetwork(virtual_circuit, results)
# result = tn.contract(all, optimize="auto")

tn.draw(color=["F", "C"])

circuit = circuit.assign_parameters(params)
counts = AerSimulator().run(circuit, shots=100000).result().get_counts()
print(abs(result - qvm.expval_from_counts(counts)))
