import pytest
from qiskit.circuit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qtpu.transforms import squash_cregs, seperate_clbits, remove_operations_by_name


def test_squash_cregs():
    # Create a quantum circuit with multiple classical registers
    qreg = QuantumRegister(2, "q")
    creg1 = ClassicalRegister(2, "c1")
    creg2 = ClassicalRegister(2, "c2")
    circuit = QuantumCircuit(qreg, creg1, creg2)

    # Add some operations to the circuit
    circuit.h(qreg[0])
    circuit.cx(qreg[0], qreg[1])
    circuit.measure(qreg[0], creg1[0])
    circuit.measure(qreg[1], creg2[1])

    # Apply the squash_cregs function
    squashed_circuit = squash_cregs(circuit)

    # Check that the squashed circuit has only one classical register
    assert len(squashed_circuit.cregs) == 1
    assert squashed_circuit.cregs[0].name == "squashed"
    assert squashed_circuit.cregs[0].size == 4

    # Check that the operations are correctly mapped to the new classical register
    assert squashed_circuit.data[2].clbits == (squashed_circuit.cregs[0][0],)
    assert squashed_circuit.data[3].clbits == (squashed_circuit.cregs[0][3],)

    # Check that the quantum operations are unchanged
    assert squashed_circuit.data[0].operation.name == "h"
    assert squashed_circuit.data[1].operation.name == "cx"

def test_seperate_clbits():
    # Create a quantum circuit with multiple classical registers
    qreg = QuantumRegister(2, "q")
    creg1 = ClassicalRegister(2, "c1")
    creg2 = ClassicalRegister(2, "c2")
    circuit = QuantumCircuit(qreg, creg1, creg2)

    # Add some operations to the circuit
    circuit.h(qreg[0])
    circuit.cx(qreg[0], qreg[1])
    circuit.measure(qreg[0], creg1[0])
    circuit.measure(qreg[1], creg2[1])

    # Apply the seperate_clbits function
    separated_circuit = seperate_clbits(circuit)

    # Check that each classical bit is in its own register
    assert len(separated_circuit.cregs) == 4
    for i, creg in enumerate(separated_circuit.cregs):
        assert creg.size == 1
        assert creg.name == f"c{i}"

    # Check that the operations are correctly mapped to the new classical registers
    assert separated_circuit.data[2].clbits == (separated_circuit.cregs[0][0],)
    assert separated_circuit.data[3].clbits == (separated_circuit.cregs[3][0],)

    # Check that the quantum operations are unchanged
    assert separated_circuit.data[0].operation.name == "h"
    assert separated_circuit.data[1].operation.name == "cx"


def test_seperate_clbits():
    # Create a quantum circuit with multiple classical registers
    qreg = QuantumRegister(2, "q")
    creg1 = ClassicalRegister(2, "c1")
    creg2 = ClassicalRegister(2, "c2")
    circuit = QuantumCircuit(qreg, creg1, creg2)

    # Add some operations to the circuit
    circuit.h(qreg[0])
    circuit.cx(qreg[0], qreg[1])
    circuit.measure(qreg[0], creg1[0])
    circuit.measure(qreg[1], creg2[1])

    # Apply the seperate_clbits function
    separated_circuit = seperate_clbits(circuit)

    # Check that each classical bit is in its own register
    assert len(separated_circuit.cregs) == 4
    for i, creg in enumerate(separated_circuit.cregs):
        assert creg.size == 1
        assert creg.name == f"c{i}"

    # Check that the operations are correctly mapped to the new classical registers
    assert separated_circuit.data[2].clbits == (separated_circuit.cregs[0][0],)
    assert separated_circuit.data[3].clbits == (separated_circuit.cregs[3][0],)

    # Check that the quantum operations are unchanged
    assert separated_circuit.data[0].operation.name == "h"
    assert separated_circuit.data[1].operation.name == "cx"


def test_remove_operations_by_name():
    # Create a quantum circuit with various operations
    qreg = QuantumRegister(2, "q")
    creg = ClassicalRegister(2, "c")
    circuit = QuantumCircuit(qreg, creg)

    # Add some operations to the circuit
    circuit.h(qreg[0])
    circuit.cx(qreg[0], qreg[1])
    circuit.measure(qreg[0], creg[0])
    circuit.measure(qreg[1], creg[1])

    # Remove measure operations by name
    modified_circuit = remove_operations_by_name(circuit, "measure", inplace=False)

    # Check that the measure operations are removed
    assert all(instr.operation.name != "measure" for instr in modified_circuit.data)

    # Check that other operations are unchanged
    assert modified_circuit.data[0].operation.name == "h"
    assert modified_circuit.data[1].operation.name == "cx"

    # Remove H and CX operations by name
    modified_circuit = remove_operations_by_name(circuit, {"h", "cx"}, inplace=False)

    # Check that the H and CX operations are removed
    assert all(
        instr.operation.name not in {"h", "cx"} for instr in modified_circuit.data
    )

    # Check that measure operations are unchanged
    assert modified_circuit.data[0].operation.name == "measure"
    assert modified_circuit.data[1].operation.name == "measure"


def test_remove_operations_by_name_complex():
    # Create a quantum circuit with various operations
    qreg = QuantumRegister(3, "q")
    creg = ClassicalRegister(3, "c")
    circuit = QuantumCircuit(qreg, creg)

    # Add some operations to the circuit
    circuit.h(qreg[0])
    circuit.cx(qreg[0], qreg[1])
    circuit.measure(qreg[0], creg[0])
    circuit.x(qreg[2])
    circuit.measure(qreg[1], creg[1])
    circuit.z(qreg[2])
    circuit.measure(qreg[2], creg[2])

    # Remove measure and X operations by name
    modified_circuit = remove_operations_by_name(
        circuit, {"measure", "x"}, inplace=False
    )

    # Check that the measure and X operations are removed
    assert all(
        instr.operation.name not in {"measure", "x"} for instr in modified_circuit.data
    )

    # Check that other operations are unchanged
    assert modified_circuit.data[0].operation.name == "h"
    assert modified_circuit.data[1].operation.name == "cx"
    assert modified_circuit.data[2].operation.name == "z"

    # Remove H, CX, and Z operations by name
    modified_circuit = remove_operations_by_name(
        circuit, {"h", "cx", "z"}, inplace=False
    )

    # Check that the H, CX, and Z operations are removed
    assert all(
        instr.operation.name not in {"h", "cx", "z"} for instr in modified_circuit.data
    )

    # Check that measure operations are unchanged
    assert modified_circuit.data[0].operation.name == "measure"
    assert modified_circuit.data[1].operation.name == "x"
    assert modified_circuit.data[2].operation.name == "measure"
    assert modified_circuit.data[3].operation.name == "measure"
