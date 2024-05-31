import numpy as np
from qiskit.circuit.library import TwoLocal
from qiskit_aer import AerSimulator
import qtpu
from qtpu.cutensor import contract_gpu
import qtpu.qinterface


N = 10
circuit = TwoLocal(N, ["ry", "rz"], "rzz", entanglement="linear", reps=2).decompose()
circuit = circuit.assign_parameters(
    {param: np.random.randn() * np.pi for param in circuit.parameters}
)
circuit.measure_all()


hybrid_tn = qtpu.cut(
    circuit, qtpu.compiler.NumQubitsOracle(N // 2), show_progress_bar=True
)

sim = AerSimulator()
# sim.set_option("cusvaer_enable", False)

result = contract_gpu(hybrid_tn, shots=100000, qiface=qtpu.qinterface.BackendInterface())

from qiskit.primitives import Estimator

print(result)
circuit.remove_final_measurements()
print(Estimator().run(circuit, observables=["Z" * N]).result().values)


from qtpu.helpers import compute_Z_expectation

print(compute_Z_expectation(circuit))
