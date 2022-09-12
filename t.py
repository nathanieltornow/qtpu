from qiskit import QuantumCircuit
from vqc.cut import cut, QubitGroups
from qiskit.providers.aer import AerSimulator


circuit = QuantumCircuit(4)
circuit.h([0, 1, 2, 3])
circuit.cx(0, 1)
circuit.cx(2, 3)
circuit.cx(1, 2)
circuit.cx(0, 1)
circuit.cx(2, 3)
circuit.measure_all()

frag1 = {circuit.qubits[0], circuit.qubits[1]}
frag2 = {circuit.qubits[2], circuit.qubits[3]}

cut_circ = cut(circuit, QubitGroups([frag1, frag2]))
print(cut_circ)
"""
         ┌───┐              ░ ┌─┐         
frag0_0: ┤ H ├──■───────■───░─┤M├─────────
         ├───┤┌─┴─┐ ░ ┌─┴─┐ ░ └╥┘┌─┐      
frag0_1: ┤ H ├┤ X ├─░─┤ X ├─░──╫─┤M├──────
         ├───┤└───┘ ░ └───┘ ░  ║ └╥┘┌─┐   
frag1_0: ┤ H ├──■───░───■───░──╫──╫─┤M├───
         ├───┤┌─┴─┐ ░ ┌─┴─┐ ░  ║  ║ └╥┘┌─┐
frag1_1: ┤ H ├┤ X ├───┤ X ├─░──╫──╫──╫─┤M├
         └───┘└───┘   └───┘ ░  ║  ║  ║ └╥┘
 meas: 4/══════════════════════╩══╩══╩══╩═
                               0  1  2  3 
"""
print(cut_circ.fragments)
for frag in cut_circ.fragments:
    print(cut_circ.fragment_as_circuit(frag))
