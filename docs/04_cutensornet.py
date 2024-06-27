import numpy as np
from qiskit.circuit.library import TwoLocal
from qiskit_aer import AerSimulator
import qtpu
from qtpu.cuquantum import contract_gpu
from qtpu.compiler.terminators import reach_num_qubits
import qtpu.qinterface


N = 10
# circuit = TwoLocal(N, ["ry", "rz"], "rzz", entanglement="linear", reps=2).decompose()
# circuit = circuit.assign_parameters(
#     {param: np.random.randn() * np.pi for param in circuit.parameters}
# )
# circuit.measure_all()
from benchmark.benchmarks import generate_benchmark

circuit = generate_benchmark("ghz", 10)


hybrid_tn = qtpu.cut(
    circuit, max_cost=16, terminate_fn=reach_num_qubits(6), show_progress_bar=True
)

sim = AerSimulator(method="statevector")
sim.set_option("cusvaer_enable", True)

for qt in hybrid_tn.quantum_tensors:
    print("--")
    for c, _ in qt.instances():
        print(c)

result = contract_gpu(hybrid_tn)

from qiskit.primitives import Estimator

print(result)
circuit.remove_final_measurements()
print(Estimator().run(circuit, observables=["Z" * N]).result().values)


from qtpu.helpers import compute_Z_expectation

print(compute_Z_expectation(circuit))
