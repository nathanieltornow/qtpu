"""Representations of quantum tensors and hybrid tensor networks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import quimb.tensor as qtn
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


class CircuitTensor:
    """A class to represent a tensor of quantum circuits.

    The tensor property comes from having InstructionVector operations in a quantum circuit.
    The circuit tensor is then the tensor product of all possible circuits that can be obtained by assigning
    different values to the parameters of the InstructionVector operations.

    Parameters
    ----------
    circuit : QuantumCircuit
        The quantum circuit associated with the tensor.

    Attributes:
    ----------
    circuit : QuantumCircuit
        The quantum circuit associated with the tensor.
    shape : tuple[int, ...]
        The shape of the tensor, representing the number of parameters for each operation.
    inds : tuple[str, ...]
        The names of the parameters in the tensor.
    """

    def __init__(self, circuit: QuantumCircuit) -> None:
        """Initialize the tensor with a given quantum circuit.

        Args:
            circuit (QuantumCircuit): The quantum circuit to initialize the tensor with.
        """
        param_to_op = list(
            {
                instr.operation.param: instr.operation
                for instr in circuit
                if isinstance(instr.operation, ISwitch)
            }.items()
        )

        self._circuit = circuit
        self._shape: tuple[int, ...] = tuple(len(op) for _, op in param_to_op)
        self._inds: tuple[str, ...] = tuple(p.name for p, _ in param_to_op)

    @property
    def circuit(self) -> QuantumCircuit:
        """Returns the QuantumCircuit object associated with this instance.

        Returns:
            QuantumCircuit: The quantum circuit associated with this instance.
        """
        return self._circuit

    @property
    def shape(self) -> tuple[int, ...]:
        """Returns the shape of the tensor.

        Returns:
            tuple[int, ...]: A tuple representing the dimensions of the tensor.
        """
        return self._shape

    @property
    def inds(self) -> tuple[str, ...]:
        """Returns the indices of the tensor (quimb-style).

        Returns:
            tuple[str, ...]: A tuple of strings representing the indices of the tensor.
        """
        return self._inds

    def __getitem__(self, index: int | tuple[int, ...]) -> QuantumCircuit:
        """Returns the circuit corresponding to the provided index.

        Args:
            index (int | tuple[int, ...]): The index of the circuit to be returned.

        Returns:
            QuantumCircuit: The circuit corresponding to the provided index.
        """
        if isinstance(index, int):
            index = (index,)

        assert len(index) == len(self.shape)

        param_map = dict(zip(self.inds, index, strict=False))
        return self._circuit.assign_parameters(param_map)

    def flat(self) -> list[QuantumCircuit]:
        """Returns the circuit tensor as a flat list.

        Returns:
            list[QuantumCircuit]: The circuit tensor flattened into a list.
        """
        indices = np.ndindex(self.shape)
        return [self[tuple(idx)] for idx in indices]


class HybridTensorNetwork:
    """A class to represent a hybrid tensor network consisting of quantum and classical tensors.

    Attributes:
    -----------
    qtensors : list[CircuitTensor]
        A list of quantum circuit tensors.
    ctensors : list[qtn.Tensor]
        A list of classical tensors.
    """

    def __init__(
        self, qtensors: list[CircuitTensor], ctensors: list[qtn.Tensor]
    ) -> None:
        """Initialize the tensor object with quantum and classical tensors.

        Args:
            qtensors (list[CircuitTensor]): A list of quantum circuit tensors.
            ctensors (list[qtn.Tensor]): A list of classical tensors.

        Returns:
            None.
        """
        self._qts = qtensors
        self._cts = ctensors

    @property
    def qtensors(self) -> list[CircuitTensor]:
        """Returns the quantum tensors in the tensor network.

        Returns:
            list[CircuitTensor]: A list of quantum tensors in the tensor network.
        """
        return self._qts

    @property
    def ctensors(self) -> list[qtn.Tensor]:
        """Returns the classical tensors in the tensor network.

        Returns:
            list[qtn.Tensor]: A list of classical tensors in the tensor network.
        """
        return self._cts

    @property
    def subcircuits(self) -> list[QuantumCircuit]:
        """Returns the circuits corresponding to the quantum tensors.

        Returns:
            list[QuantumCircuit]: A list of quantum circuits corresponding to the quantum tensors.
        """
        return [qt.circuit for qt in self._qts]

    def to_dummy_tn(self) -> qtn.TensorNetwork:
        """Convert the current object to a dummy classical tensor network.

        To be used for analysis purposes.

        Returns:
            qtn.TensorNetwork: A tensor network consisting of dummy quantum tensors
                and the original classical tensors.
        """
        qt_dummys = [
            qtn.Tensor(np.zeros(qt.shape), inds=qt.inds, tags=["Q"]) for qt in self._qts
        ]
        return qtn.TensorNetwork(qt_dummys + self._cts)
