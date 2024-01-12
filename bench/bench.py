
from qvm.runtime.runner import expval_from_counts, sample_fragments
from qvm.runtime.runners import SimRunner
from qvm.runtime.virtualizer import build_tensornetwork, build_dummy_tensornetwork
from qvm.cutter.girvan_newman import GirvanNewmanCutter
from qvm.virtual_circuit import VirtualCircuit
from qiskit_aer import AerSimulator

from circuits import get_circuits

circuit = get_circuits("qaoa_ba1", (10, 11))[0]

cutter = GirvanNewmanCutter(100)
cut_circuit = cutter.run(circuit)
print(cut_circuit)

virtual_circuit = VirtualCircuit(cut_circuit)
print(virtual_circuit.num_instantiations())

# tn = build_dummy_tensornetwork(virtual_circuit)
# print(tn.contraction_cost(optimize="auto"))
# tn.draw(color=["frag_result", "coeff"])


results = sample_fragments(virtual_circuit, SimRunner(), shots=100000)
tn = build_tensornetwork(virtual_circuit, results)
result = tn.contract(all, optimize="auto")

counts = AerSimulator().run(circuit, shots=100000).result().get_counts()
print(abs(result - expval_from_counts(counts)))
