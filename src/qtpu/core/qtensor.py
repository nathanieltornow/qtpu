"""QuantumTensor class for representing tensors of quantum circuits."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qiskit.circuit import QuantumCircuit, Parameter

from qtpu.core.iswitch import ISwitch

if TYPE_CHECKING:
    from qtpu.compiler.codegen import CompiledQuantumTensor


class QuantumTensor:
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

    def compile(self, warmup: bool = True) -> "CompiledQuantumTensor":
        """Compile this quantum tensor for fast repeated evaluation.

        Compiles the quantum circuit to CUDA-Q for efficient execution.
        The compiled tensor caches the kernel, so the first call includes
        JIT compilation overhead but subsequent calls are fast.

        Returns:
            CompiledQuantumTensor: A compiled tensor with a fast __call__ method.

        Example:
            >>> qtensor = QuantumTensor(circuit)
            >>> compiled = qtensor.compile()
            >>>
            >>> # First call includes compilation
            >>> result = compiled()  # shape matches qtensor.shape
            >>>
            >>> # Subsequent calls are fast (kernel cached)
            >>> result = compiled(theta=0.5, phi=1.2)  # with parameters
        """
        from qtpu.compiler.codegen import CompiledQuantumTensor

        return CompiledQuantumTensor(self, warmup=warmup)

    @classmethod
    def from_shape(
        cls,
        shape: tuple[int, ...],
        inds: tuple[str, ...],
        base_circuit: QuantumCircuit | None = None,
    ) -> "QuantumTensor":
        """Create a QuantumTensor with the specified shape and indices.

        Creates a quantum circuit with ISwitches to achieve the desired tensor
        shape. Each index corresponds to an ISwitch with the corresponding
        dimension.

        Args:
            shape: The desired shape of the tensor.
            inds: The index names (must match length of shape).
            base_circuit: Optional base circuit to use. If None, uses a 10-qubit
                QNN circuit from mqt.bench as the default.

        Returns:
            QuantumTensor: A quantum tensor with the specified shape.

        Raises:
            ValueError: If shape and inds have different lengths.

        Example:
            >>> qt = QuantumTensor.from_shape((3, 4), ("i", "j"))
            >>> qt.shape
            (3, 4)
            >>> qt.inds
            ('i', 'j')
        """
        if len(shape) != len(inds):
            raise ValueError(
                f"Shape and inds must have same length: {len(shape)} vs {len(inds)}"
            )

        if base_circuit is None:
            from mqt.bench import get_benchmark_indep

            base_circuit = get_benchmark_indep("qnn", 10)

        num_qubits = base_circuit.num_qubits
        qc = base_circuit.copy()

        # Add classical register if not present
        if qc.num_clbits == 0:
            from qiskit.circuit import ClassicalRegister

            qc.add_register(ClassicalRegister(num_qubits))

        # Add an ISwitch for each index
        for i, (ind_name, dim) in enumerate(zip(inds, shape, strict=True)):
            param = Parameter(ind_name)

            def make_identity(k: int) -> QuantumCircuit:
                """Create an identity circuit (no operation)."""
                return QuantumCircuit(1)

            iswitch = ISwitch(param, 1, dim, make_identity)
            qc.append(iswitch, [i % num_qubits])

        # Add measurement if not present
        if not any(instr.operation.name == "measure" for instr in qc):
            qc.measure(range(num_qubits), range(num_qubits))

        return cls(qc)
