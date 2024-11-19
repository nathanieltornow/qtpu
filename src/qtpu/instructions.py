import numpy as np
from qiskit.circuit import QuantumCircuit, Instruction, Parameter


class InstructionVector(Instruction):
    def __init__(
        self, instructions_vector: list[list[Instruction]], idx_param: Parameter
    ):
        assert all(
            instr.num_qubits == 1 for instrs in instructions_vector for instr in instrs
        )
        assert isinstance(idx_param, Parameter)

        super().__init__("vec", 1, 0, params=(idx_param,))
        self._vector = instructions_vector

    @property
    def param(self) -> Parameter:
        return self.params[0]

    @property
    def vector(self) -> list[list[Instruction]]:
        return self._vector

    def __len__(self):
        return len(self.vector)

    def _define(self):
        circuit = QuantumCircuit(1, 0)
        param_value = int(self.params[0])

        for instr in self.vector[param_value]:
            circuit.append(instr, [0])
        self.definition = circuit

    