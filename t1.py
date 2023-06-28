from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile


from bench._circuits import get_circuits


N = 8
circuit = QuantumCircuit(N)
circuit.h(range(N))

circuit.cx(range(N - 1), range(1, N))

circuit.cx(range(N - 1), range(1, N))
circuit.cx(range(N - 1), range(1, N))
circuit.h(range(N))
# circuit.cx(range(N-1), range(1, N))
# circuit.cx(0, N-1)

circuit = get_circuits("hamsim", 3, nums_qubits=[N])[0]

from qvm.compiler.virtualization.reduce_swap import ReduceSWAPCompiler
from qvm.compiler.virtualization.general_bisection import GeneralBisectionCompiler

from qiskit.providers.fake_provider import FakeMumbaiV2

# comp = ReduceSWAPCompiler(FakeMumbaiV2(), max_virtual_gates=2, reverse_order=False, max_distance=3)
comp = GeneralBisectionCompiler(max_virtual_gates=3, reverse_order=True)

t = comp.run(circuit)
while len(t.qregs) > 1:
    t = comp.run(circuit)

print(t)
