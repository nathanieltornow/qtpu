from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile


from bench._circuits import get_circuits


circuit = get_circuits("2local", 2, nums_qubits=[8])[0]

from qvm.compiler.virtualization.reduce_swap import ReduceSWAPCompiler
from qvm.compiler.virtualization.general_bisection import GeneralBisectionCompiler

from qiskit.providers.fake_provider import FakeMumbaiV2

comp = ReduceSWAPCompiler(FakeMumbaiV2(), max_virtual_gates=2, reverse_order=False)

print(comp.run(circuit))
