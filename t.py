from qiskit import (
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    execute,
    transpile,
)
from qiskit.circuit.quantumcircuit import Qubit, Instruction, CircuitInstruction
from qiskit.circuit.library.standard_gates import CZGate

from qvm.circuit import VirtualCircuit, FragmentedVirtualCircuit
from qvm.circuit.virtual_gate.virtual_cz import VirtualCZ
from qvm.execution.execute import execute_virtual_circuit
from qvm.bench import fidelity
from qiskit.providers.aer import AerSimulator


circuit = QuantumCircuit(3, 3)
circuit.h(0)
circuit.h(1)
circuit.cx(0, 1)
circuit.cx(1, 2)
circuit.measure(0, 0)
circuit.measure(1, 1)


# circuit.measure_all()
print(AerSimulator().run(circuit, shots=20000).result().get_counts())

cp_circ = circuit.copy()
print(circuit)


frag_circ = FragmentedVirtualCircuit.from_circuit(circuit)
frag_circ.virtualize_connection(frag_circ.qubits[0], frag_circ.qubits[1])
frag_circ.virtualize_connection(frag_circ.qubits[1], frag_circ.qubits[2])

print("--------------------------------------------")

frag_circ.create_fragments()
for frag in frag_circ.fragments:
    print(frag.base_circuit)

# vcirc.append(VirtualCZ(CZGate()), [vcirc.qubits[0], vcirc.qubits[1]])

print("--------------------------------------------")

# frag_circ.cx(2, 1)


frag_circ.create_fragments()
for frag in frag_circ.fragments:
    print(frag.base_circuit)

# vcirc.virtualize_connection(vcirc.qubits[1], vcirc.qubits[2])
print(frag_circ.graph.edges)
print(frag_circ.fragment_virtual_gates)


# result = execute_virtual_circuit(vcirc, backend=AerSimulator())

# print(result.counts())
# print(fidelity(cp_circ, result.counts()))

# for conf, circ in vcirc.configured_circuits().items():
#     print(conf)
#     print(circ.decompose())
#     print()

# import time
# from itertools import combinations
# from qiskit.compiler import assemble
# from qiskit.test.mock import FakeVigo
# from qiskit.circuit import Parameter
# import numpy as np

# theta_range = np.linspace(0, 2 * np.pi, 128)
# start = time.time()
# qc = QuantumCircuit(5)
# theta = Parameter("theta")
# alpha = Parameter("alpha")

# for k in range(8):
#     # for i,j in combinations(range(5), 2):
#     #     qc.cx(i,j)
#     qc.rz(theta, range(5))
#     qc.ry(alpha, range(5))
#     # for i,j in combinations(range(5), 2):
#     #     qc.cx(i,j)

# print(qc.parameters)

# transpiled_qc = transpile(qc, backend=FakeVigo())
# print(transpiled_qc)
# qobj = assemble(
#     [transpiled_qc.bind_parameters({theta: n, alpha: n}) for n in theta_range],
#     backend=FakeVigo(),
# )
# end = time.time()
# print("Time compiling over parameterized circuit, then binding: ", end - start)


# from qiskit.result.models import ExperimentResult
# # Build a sub-circuit
# sub_circ = QuantumCircuit(2, 2,  name='sub_circ')
# sub_circ.h(0)
# sub_circ.crz(1, 0,1)
# sub_circ.barrier()
# sub_circ.id(1)
# sub_circ.u(1, 2, -2, 0)
# print(sub_circ)


# # Convert to a gate and stick it into an arbitrary place in the bigger circuit
# sub_inst = sub_circ.to_instruction(label='sub_inst2')
# print("hi, \n", type(sub_inst.definition))

# circ = QuantumCircuit(3, 3)
# circ.h(0)
# circ.cx(0, 1)
# circ.cx(1, 2)

# print(circ)
# circ._data[2] = CircuitInstruction(sub_inst, [circ.qubits[1], circ.qubits[2]], [circ.clbits[0], circ.clbits[1]])
# # circ.append(sub_inst, [1, 2], [1, 2])
# print(circ)

# print(sub_inst == Instruction("a", 1,1, []))
