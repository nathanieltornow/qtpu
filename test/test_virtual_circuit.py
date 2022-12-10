import unittest

from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister

from vqc.virtual_circuit import VirtualCircuit, AbstractCircuit, Placeholder
from vqc.virtual_gates import VirtualCX, VirtualCZ


class TestVirtualCircuit(unittest.TestCase):
    def _example_circuit(self) -> QuantumCircuit:
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.y(2)
        circuit.append(VirtualCX(), [0, 2], [])
        circuit.append(VirtualCZ(), [1, 2], [])
        circuit.measure_all()
        return circuit

    def _abstr_circ0(self) -> AbstractCircuit:
        qreg = QuantumRegister(2, "frag0")
        meas = ClassicalRegister(3, "meas")
        conf_reg = ClassicalRegister(2, "conf")
        circuit = AbstractCircuit(qreg, meas, conf_reg)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.append(Placeholder("config_0_0"), (0,), (conf_reg[0],))
        circuit.append(Placeholder("config_1_0"), (1,), (conf_reg[1],))
        circuit.measure(0, 0)
        circuit.measure(1, 1)
        return circuit

    def _abstr_circ1(self) -> AbstractCircuit:
        qreg = QuantumRegister(1, "frag1")
        meas = ClassicalRegister(3, "meas")
        conf_reg = ClassicalRegister(2, "conf")
        circuit = AbstractCircuit(qreg, meas, conf_reg)
        circuit.y(0)
        circuit.append(Placeholder("config_0_1"), (0,), (conf_reg[0],))
        circuit.append(Placeholder("config_1_1"), (0,), (conf_reg[1],))
        circuit.measure(0, 2)
        return circuit

    def test_fragments(self):
        circuit = self._example_circuit()
        virtual_circuit = VirtualCircuit(circuit)
        self.assertEqual(len(virtual_circuit.fragments), 2)
        frag0, frag1 = virtual_circuit.fragments
        self.assertEqual(len(frag0), 2)
        self.assertEqual(frag0.name, "frag0")
        self.assertEqual(len(frag1), 1)
        self.assertEqual(frag1.name, "frag1")

    def test_virtual_gates(self):
        circuit = self._example_circuit()
        virtual_circuit = VirtualCircuit(circuit)
        self.assertEqual(len(virtual_circuit.virtual_gates), 2)
        self.assertEqual(type(virtual_circuit.virtual_gates[0]), VirtualCX)
        self.assertEqual(type(virtual_circuit.virtual_gates[1]), VirtualCZ)

    def test_is_valid(self):
        circuit = self._example_circuit()
        virtual_circuit = VirtualCircuit(circuit)
        self.assertTrue(virtual_circuit.is_valid)
        virtual_circuit.cx(0, 2)
        self.assertFalse(virtual_circuit.is_valid)

    def test_config_ids(self):
        circuit = self._example_circuit()
        virtual_circuit = VirtualCircuit(circuit)
        frag0, frag1 = virtual_circuit.fragments
        self.assertEqual(len(list(virtual_circuit._config_ids(frag0))), 36)
        self.assertEqual(len(list(virtual_circuit._config_ids(frag1))), 36)

    def test_abstract_circuit(self):
        circuit = self._example_circuit()
        virtual_circuit = VirtualCircuit(circuit)
        frag0, frag1 = virtual_circuit.fragments
        abstr_circ0 = virtual_circuit.abstract_circuit(frag0)
        abstr_circ1 = virtual_circuit.abstract_circuit(frag1)
        self.assertTrue(abstr_circ0 == self._abstr_circ0())
        self.assertTrue(abstr_circ1 == self._abstr_circ1())
