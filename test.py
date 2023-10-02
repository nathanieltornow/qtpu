from qiskit.circuit import QuantumCircuit
from qiskit.circuit.random import random_circuit

import qvm

from examples.fid import calculate_fidelity


def random_gate(num_qubits):
    return random_circuit(num_qubits, num_qubits, measure=False).to_gate()


def main():
    circuit = QuantumCircuit(7, 7)
    circuit.append(random_gate(3), range(3))

    circuit.append(random_gate(2), [3, 4])
    circuit.cx(4, 5)
    circuit.append(random_gate(2), [5, 6])

    circuit.measure(range(7), range(7))

    circuit = circuit.decompose()
    print(circuit)

    comp = qvm.CutterCompiler(size_to_reach=3)
    virtual_circuit = comp.run(circuit, budget=2)
    for frag in virtual_circuit.fragment_circuits.values():
        print(frag)
    result, _ = qvm.run_virtual_circuit(virtual_circuit, shots=10000)
    print(calculate_fidelity(circuit, result))


if __name__ == "__main__":
    main()
