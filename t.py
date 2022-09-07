from time import time
from qiskit import (
    QuantumCircuit,
)
from qiskit.circuit.quantumcircuit import Qubit, Instruction, CircuitInstruction
from qiskit.circuit.library.standard_gates import CZGate
from qvm.execution.exec import execute_fragmented_circuit

from qvm.circuit import VirtualCircuit, Fragment
from qvm.bench.fidelity import fidelity
from qiskit.providers.aer import AerSimulator
from qiskit.circuit.random import random_circuit

from qvm.execution.configurator import Configurator, compute_virtual_gate_info

from qvm.bench.fidelity import fidelity

from cloudpickle import dump, load

from qvm.virtual_gate.virtual_gate import VirtualBinaryGate


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
circuit.cx(0, 1)
circuit.rzz(0.3, 2, 1)
# circuit.rzz(1.2, 2, 1)
circuit.x(2)
circuit.s(1)
circuit.cz(2, 3)
circuit.cz(3, 4)
circuit.h(2)
# circuit.measure(0, 0)
# circuit.measure(1, 1)
# circuit.measure(2, 2)
# circuit.measure(3, 3)
# circuit.measure(4, 4)
circuit.measure_all()
circuit.measure_active()

circuit.compose(random_circuit(5, 2, 1, True), inplace=True)
circ_cp = circuit.copy()


from qvm.cut.transpiler import virtualize_connection

virtualize_connection(circuit, circuit.qubits[1], circuit.qubits[2])
virtualize_connection(circuit, circuit.qubits[3], circuit.qubits[4])

print(circuit)
# print(circuit.decompose([VirtualBinaryGate,]))
# # exit(0)
frag_circ = VirtualCircuit(
    circuit.decompose(
        [
            VirtualBinaryGate,
        ]
    )
)
# print(frag_circ)


# for frag in frag_circ.fragments:

#     conf = Configurator(frag, compute_virtual_gate_info(frag_circ))
#     for i, circ in conf.configured_circuits():
#         print(i)
#         print(circ.decompose(["conf"]))

counts = execute_fragmented_circuit(frag_circ)

# for config in conf:
#     print('hi')
#     print(config.decompose())
print(fidelity(circ_cp, counts))
