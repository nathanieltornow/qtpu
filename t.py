from pytket import Circuit
from pytket.pauli import Pauli, QubitPauliString


c = Circuit()
q_reg = c.add_q_register("a", 1)
c.Sdg(q_reg[0])
# c.H(0, condition_bits=[0], condition_value=1)


ps = QubitPauliString(c.qubits[0], Pauli.Z)
print(c.qubits)
p = Pauli(1)
print(p)
print(c.get_commands())