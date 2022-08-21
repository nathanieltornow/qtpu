from dis import Instruction
from typing import Dict, List, Set
from qiskit.circuit.quantumcircuit import (
    QuantumCircuit,
    CircuitInstruction,
    Qubit,
    Clbit,
    Bit,
)
from qiskit.converters import circuit_to_dag


def mapped_qubits(
    origin: QuantumCircuit, dest: QuantumCircuit, qubits: List[Qubit]
) -> List[Qubit]:
    return [dest.qubits[origin.find_bit(q).index] for q in qubits]


def mapped_clbits(
    origin: QuantumCircuit, dest: QuantumCircuit, clbits: List[Clbit]
) -> List[Clbit]:
    return [dest.clbits[origin.find_bit(c).index] for c in clbits]


def mapped_instruction(
    origin: QuantumCircuit, dest: QuantumCircuit, instr: CircuitInstruction
) -> CircuitInstruction:
    return CircuitInstruction(
        instr.operation,
        mapped_qubits(origin, dest, instr.qubits),
        mapped_clbits(origin, dest, instr.clbits),
    )


def bit_index(circuit: QuantumCircuit, bit: Bit) -> int:
    return circuit.find_bit(bit).index


def bit_indices(circuit: QuantumCircuit, bits: List[Bit]) -> List[int]:
    return [circuit.find_bit(b).index for b in bits]


def deflated_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    idle_qubits = set(
        bit for bit in circuit_to_dag(circuit).idle_wires() if isinstance(bit, Qubit)
    )
    new_circuit = QuantumCircuit(
        circuit.num_qubits - len(idle_qubits), circuit.num_clbits
    )
    sorted_qubits = sorted(
        set(circuit.qubits) - idle_qubits, key=lambda q: circuit.find_bit(q).index
    )
    qubit_map: Dict[Qubit, Qubit] = {
        q: new_circuit.qubits[i] for i, q in enumerate(sorted_qubits)
    }
    for instr in circuit.data:
        new_circuit.append(
            instr.operation,
            [qubit_map[q] for q in instr.qubits],
            mapped_clbits(circuit, new_circuit, instr.clbits),
        )
    return new_circuit


def circuit_on_qubits(
    circuit: QuantumCircuit, qubits: Set[Qubit], deflated: bool = False
) -> QuantumCircuit:
    new_circuit = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
    for instr in circuit.data:
        if set(instr.qubits) <= qubits:
            new_circuit.append(
                mapped_instruction(circuit, new_circuit, instr),
            )
    if deflated:
        return deflated_circuit(new_circuit)
    return new_circuit
