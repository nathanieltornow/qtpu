from qiskit import QuantumCircuit, IBMQ, transpile

from qiskit.circuit.quantumcircuit import Qubit, Instruction, CircuitInstruction

import lithops


def f(x):
    return x + 1


fexec = lithops.FunctionExecutor()

fexec.call_async(f, 3)
fexec.call_async(f, 4)

print(fexec.get_result())
print(fexec.get_result())
