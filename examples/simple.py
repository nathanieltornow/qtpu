import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import TwoLocal, QuantumVolume
from qiskit_aer import AerSimulator

# from qvm.runtime.runner import expval_from_counts, sample_fragments
# from qvm.runtime.runners import SimRunner
# from qvm.tn import build_tensornetwork, build_dummy_tensornetwork
# from qvm.cutters.greedy import cut_greedily
# from qvm.cutters.success_estimator import QPUSizeEstimator, InstanceCostEstimator
from qvm.cut.cut import cut_central_edges

# from qvm.virtual_circuit import VirtualCircuit
import qvm


circuit = TwoLocal(
    5,
    rotation_blocks=["rz", "ry"],
    entanglement_blocks="rzz",
    entanglement="linear",
    reps=2,
).decompose()
circuit.measure_all()


params = {param: np.random.randn() / 2 for param in circuit.parameters}

# cut_circuit = GirvanNewmanCutter(100).run(circuit)
# cut_circuit = MetisCutter(3).run(circuit)


cut_circuit = cut_central_edges(circuit, 2)

print(cut_circuit)

exit()

cut_circuit = cut_circuit.assign_parameters(params)

virtual_circuit = qvm.VirtualCircuit(cut_circuit)
print(virtual_circuit.num_instantiations())

result = qvm.run_virtual_circuit(virtual_circuit, shots=100000)

# tn = build_dummy_tensornetwork(virtual_circuit)
# print(tn.contraction_cost(optimize="auto"))


# results = sample_fragments(virtual_circuit, SimRunner(), shots=100000)
# tn = build_tensornetwork(virtual_circuit, results)
# result = tn.contract(all, optimize="auto")

# tn.draw(color=["F", "C"])

circuit = circuit.assign_parameters(params)
counts = AerSimulator().run(circuit, shots=100000).result().get_counts()
print(abs(result - qvm.expval_from_counts(counts)))
