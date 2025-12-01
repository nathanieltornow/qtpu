"""ISwitch instruction for parameterized quantum circuit selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qiskit.circuit import Instruction, Parameter, QuantumCircuit

if TYPE_CHECKING:
    from collections.abc import Callable


class ISwitch(Instruction):
    """ISwitch instruction for quantum tensors."""

    def __init__(
        self,
        idx_param: Parameter,
        num_qubits: int,
        size: int,
        selector: Callable[[int], QuantumCircuit],
    ) -> None:
        """Initialize the ISwitch instruction.

        Args:
            idx_param (Parameter): The parameter used to index the ISwitch instruction.
            num_qubits (int): The number of qubits in the ISwitch instruction.
            size (int): The size of the ISwitch instruction.
            selector (Callable[[int], QuantumCircuit]): A function that selects a circuit
                based on the index parameter.
        """
        self._size = size
        self._selector = selector
        super().__init__("iswitch", num_qubits, 0, params=(idx_param,))

    @property
    def size(self) -> int:
        """Returns the size of the ISwitch instruction.

        Returns:
            int: The size of the ISwitch instruction.
        """
        return self._size

    @property
    def param(self) -> Parameter:
        """Returns the parameter used to index the ISwitch instruction.

        Returns:
            Parameter: The parameter used to index the ISwitch instruction.
        """
        return self.params[0]

    def __len__(self) -> int:
        """Returns the size of the ISwitch instruction.

        Returns:
            int: The size of the ISwitch instruction.
        """
        return self._size

    @staticmethod
    def from_1q_instructions(
        idx_param: Parameter, instructions: list[list[Instruction]]
    ) -> ISwitch:
        """Create an ISwitch instruction from a list of instruction lists.

        Args:
            idx_param (Parameter): The parameter used to index the ISwitch instruction.
            instructions (list[list[Instruction]]): A list of instruction lists.

        Returns:
            ISwitch: The created ISwitch instruction.

        Raises:
            ValueError: If the instruction lists do not have the same number of qubits.
        """
        size = len(instructions)

        def _selector(index: int) -> QuantumCircuit:
            circuit = QuantumCircuit(1)
            for instr in instructions[index]:
                circuit.append(instr, (0,), ())
            return circuit

        return ISwitch(idx_param, 1, size, _selector)

    def _define(self) -> None:
        param_value = int(self.params[0])

        if param_value < 0 or param_value >= self.size:
            msg = f"Parameter value {param_value} out of bounds for ISwitch of size {self.size}."
            raise ValueError(msg)

        selected_circuit = self._selector(param_value)

        if selected_circuit.num_qubits != self.num_qubits:
            msg = (
                f"Selected circuit has {selected_circuit.num_qubits} qubits, "
                f"but ISwitch expects {self.num_qubits} qubits."
            )
            raise ValueError(msg)

        if selected_circuit.num_clbits != 0:
            msg = (
                f"Selected circuit has {selected_circuit.num_clbits} classical bits, "
                "but ISwitch expects 0 classical bits."
            )
            raise ValueError(msg)

        self._definition = selected_circuit
