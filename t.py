from time import time
from qiskit import (
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    execute,
    transpile,
)
from qiskit.circuit.quantumcircuit import Qubit, Instruction, CircuitInstruction
from qiskit.circuit.library.standard_gates import CZGate

from qvm.circuit import VirtualCircuit
from qvm.circuit.virtual_circuit import Fragment
from qvm.circuit.virtual_gate.virtual_cz import VirtualCZ
from qvm.bench.fidelity import fidelity
from qiskit.providers.aer import AerSimulator
from qvm.circuit.virtual_gate.virtual_gate import VirtualBinaryGate
from qvm.execution.exec import execute_fragmented_circuit, execute_virtual_circuits
from qiskit.circuit.random import random_circuit

from qvm.execution.knit import chunk

from cloudpickle import dump, load

from qvm.transpiler.transpiled_fragment import TranspiledVirtualCircuit

# l = [1, 2, 3, 4]

# for c in chunk(l, 2):
#     print(list(c))

# exit(0)
# while True:
# with open("circuit.pickle", "rb") as f:
circuit = QuantumCircuit(5, 5)
circuit.compose(random_circuit(5, 3, 1, False), inplace=True)
circuit.h(0)
circuit.h(1)
# circuit.cx(0, 1)
circuit.rzz(0.3, 2, 1)
circuit.rzz(1.2, 2, 1)
circuit.x(2)
circuit.s(1)
circuit.cz(2, 3)
# circuit.cz(3, 4)
circuit.h(2)
# # circuit.measure_active()

circuit.compose(random_circuit(5, 2, 1, True), inplace=True)
circ_cp = circuit.copy()

vc = VirtualCircuit.from_circuit(circuit)

vc.virtualize_connection(circuit.qubits[1], circuit.qubits[2])
vc.virtualize_connection(circuit.qubits[4], circuit.qubits[3])
print(vc)

frag = vc.fragments.pop()

# for conf_id, circ in FragmentConfigurator(vc, frag).configured_circuits():
#     print(conf_id)
#     print(circ)

# for conf_id, circ in BinaryFragmentsConfigurator(vc):
#     print(conf_id)
#     print(circ[0])
#     print(circ[1])
# if sum(1 for instr in vc.data if isinstance(instr.operation, VirtualBinaryGate)) > 3:
#     print("Too many virtual gates")

print(len(vc.fragments))

begin = time()
counts = execute_fragmented_circuit(TranspiledVirtualCircuit(vc)).counts(50000)
print(time() - begin)

# # begin = time()
# # counts = execute_virtual_circuit([vc])[0].counts(50000)
# # print(time() - begin)

print(counts)
fid = fidelity(circ_cp, counts)
print(fid)
# if fid < 0.99:
#     break

# with open("circuit.pickle", "wb") as f:
#     dump(circ_cp, f)

# vcircuit = VirtualCircuit(circuit.copy())
# vcircuit.virtualize_connection(0, 1)
# vcircuit.virtualize_connection(1, 2)


# frag_tree = FragmentTree(vcircuit.circuit(), set(circuit.qubits))

# frag_tree.create_fragments(set(circuit.qubits[:1]), set(circuit.qubits[1:]))

# frags = frag_tree.fragments()
# if frags:
#     frag1, frag2 = frags

#     print(frag2.circuit())

#     for _, circ in frag2.configured_circuits():
#         print(circ.circuit())
#         print(circ.circuit().decompose())
