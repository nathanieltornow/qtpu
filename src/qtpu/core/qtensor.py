"""QuantumTensor and CompiledQuantumTensor classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qiskit.circuit import QuantumCircuit, Parameter

from qtpu.core.iswitch import ISwitch

if TYPE_CHECKING:
    pass


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

    def compile(self, backend: str = "cudaq") -> "CompiledQuantumTensor":
        """Compile this quantum tensor for fast repeated evaluation.

        Compiles the quantum circuit to a native representation that can be
        executed efficiently. The compiled tensor caches the kernel, so the
        first call includes JIT compilation overhead but subsequent calls
        are fast.

        Args:
            backend: Compilation backend. Currently supported:
                - "cudaq": NVIDIA CUDA-Q (requires cuda-quantum package)

        Returns:
            CompiledQuantumTensor: A compiled tensor with a fast __call__ method.

        Raises:
            ValueError: If the backend is not supported.

        Example:
            >>> qtensor = QuantumTensor(circuit)
            >>> compiled = qtensor.compile("cudaq")
            >>> 
            >>> # First call includes compilation
            >>> result = compiled()  # shape matches qtensor.shape
            >>> 
            >>> # Subsequent calls are fast (kernel cached)
            >>> result = compiled(theta=0.5, phi=1.2)  # with parameters
        """
        if backend == "cudaq":
            return CompiledQuantumTensor(self, backend="cudaq")
        else:
            raise ValueError(
                f"Unknown backend: {backend}. Supported: 'cudaq'"
            )

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


class CompiledQuantumTensor:
    """A compiled quantum tensor for fast repeated evaluation.

    This class wraps a QuantumTensor with a compiled backend (e.g., CUDA-Q)
    for efficient repeated execution. The kernel is JIT-compiled on first
    use and cached for subsequent calls.

    The compiled tensor is callable and returns a numpy array matching
    the original tensor's shape.

    Attributes:
        qtensor: The original QuantumTensor.
        shape: Shape of the output tensor.
        inds: Index names of the output tensor.
        backend: The compilation backend used.

    Example:
        >>> compiled = qtensor.compile("cudaq")
        >>> result = compiled()  # Returns np.ndarray with shape qtensor.shape
        >>> result = compiled(theta=0.5)  # Pass rotation parameters
    """

    def __init__(self, qtensor: QuantumTensor, backend: str = "cudaq"):
        """Initialize a compiled quantum tensor.

        Args:
            qtensor: The QuantumTensor to compile.
            backend: Compilation backend ("cudaq").
        """
        self._qtensor = qtensor
        self._backend = backend
        self._compiled_fn: callable | None = None
        self._free_param_names: list[str] = []

    @property
    def qtensor(self) -> QuantumTensor:
        """The original QuantumTensor."""
        return self._qtensor

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the output tensor."""
        return self._qtensor.shape

    @property
    def inds(self) -> tuple[str, ...]:
        """Index names of the output tensor."""
        return self._qtensor.inds

    @property
    def backend(self) -> str:
        """The compilation backend."""
        return self._backend

    @property
    def is_compiled(self) -> bool:
        """Whether the kernel has been compiled."""
        return self._compiled_fn is not None

    def _ensure_compiled(self) -> None:
        """Compile the kernel if not already compiled."""
        if self._compiled_fn is not None:
            return

        if self._backend == "cudaq":
            from qtpu.runtime.cudaq_converter import compile_cudaq_kernel
            _, self._compiled_fn, self._free_param_names = compile_cudaq_kernel(
                self._qtensor.circuit, self._qtensor.shape
            )
        else:
            raise ValueError(f"Unknown backend: {self._backend}")

    def __call__(self, **params: float) -> np.ndarray:
        """Evaluate the compiled quantum tensor.

        Args:
            **params: Values for free parameters (rotation angles, etc.).
                ISwitch parameters are handled internally.

        Returns:
            np.ndarray: Result tensor with shape matching self.shape.

        Example:
            >>> result = compiled()  # No free parameters
            >>> result = compiled(theta=0.5, phi=1.2)  # With parameters
        """
        self._ensure_compiled()

        # Build kwargs for the compiled function
        kwargs = {}
        for name in self._free_param_names:
            if name in params:
                kwargs[name] = params[name]
            else:
                # Try to find parameter with original name (e.g., 'theta[0]' vs 'theta_0')
                from qtpu.runtime.cudaq_converter import _sanitize_param_name
                for orig_name, val in params.items():
                    if _sanitize_param_name(orig_name) == name:
                        kwargs[name] = val
                        break
                else:
                    if name not in kwargs:
                        raise ValueError(
                            f"Missing parameter: '{name}'. "
                            f"Required: {self._free_param_names}"
                        )

        if kwargs:
            return self._compiled_fn(**kwargs)
        else:
            return self._compiled_fn()

    def clear_cache(self) -> None:
        """Clear the compiled kernel cache, forcing recompilation on next call."""
        self._compiled_fn = None
        self._free_param_names = []

    def __repr__(self) -> str:
        status = "compiled" if self.is_compiled else "not compiled"
        return (
            f"CompiledQuantumTensor(shape={self.shape}, "
            f"backend='{self._backend}', {status})"
        )
