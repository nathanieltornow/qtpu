from qiskit.circuit import QuantumCircuit

from qvm.instructions import VirtualBinaryGate


def num_virtual_gates(circuit: QuantumCircuit) -> int:
    return sum(1 for instr in circuit if isinstance(instr.operation, VirtualBinaryGate))


# def _wire_cuts_to_moves(circuit: QuantumCircuit) -> None:
#         move_reg = QuantumRegister(num_wire_cuts, "vmove")
#         dag.add_qreg(move_reg)

#         qubit_mapping: dict[Qubit, Qubit] = {}

#         def _find_qubit(qubit: Qubit) -> Qubit:
#             while qubit in qubit_mapping:
#                 qubit = qubit_mapping[qubit]
#             return qubit

#         cut_ctr = 0
#         for node in nx.topological_sort(dag):
#             instr = dag.get_node_instr(node)
#             instr.qubits = [_find_qubit(qubit) for qubit in instr.qubits]
#             if isinstance(instr.operation, WireCut):
#                 instr.operation = VirtualMove(SwapGate())
#                 instr.qubits.append(move_reg[cut_ctr])
#                 qubit_mapping[instr.qubits[0]] = instr.qubits[1]
#                 cut_ctr += 1