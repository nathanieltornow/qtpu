"""Hybrid einsum API for specifying tensor network contractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import cotengra as ctg

from qtpu.core.iswitch import ISwitch
from qtpu.core.qtensor import QuantumTensor
from qtpu.core.ctensor import CTensor

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


@dataclass
class TensorSpec:
    """Lightweight specification of a tensor's shape and indices."""
    shape: tuple[int, ...]
    inds: tuple[str, ...]


class HEinsum:
    """High-level API for specifying hybrid tensor network contractions."""

    def __init__(
        self,
        qtensors: list[QuantumTensor],
        ctensors: list[CTensor],
        input_tensors: list[TensorSpec],
        output_inds: tuple[str, ...],
    ):
        """Initialize a hybrid einsum specification.

        Args:
            qtensors: Quantum circuit tensors.
            ctensors: Classical tensors.
            input_tensors: Input tensor specifications (provided at runtime).
            output_inds: Output indices for the contraction result.

        Raises:
            ValueError: If indices are inconsistent or output indices not found.
        """
        self._qtensors = qtensors
        self._ctensors = ctensors
        self._input_tensors = input_tensors
        self._output_inds = output_inds

        # Build einsum expression from tensor specs
        ind_to_char = {}
        inputs = ""
        ind_sizes = {}

        next_char = ord("a")
        for tensor in qtensors + ctensors + input_tensors:
            input_entry = ""
            for i, ind in enumerate(tensor.inds):
                if ind not in ind_to_char:
                    ind_to_char[ind] = chr(next_char)
                    ind_sizes[ind] = tensor.shape[i]
                    next_char += 1
                else:
                    # Validate consistent sizes
                    if ind_sizes[ind] != tensor.shape[i]:
                        raise ValueError(
                            f"Index '{ind}' has inconsistent sizes: "
                            f"{ind_sizes[ind]} vs {tensor.shape[i]}"
                        )

                input_entry += ind_to_char[ind]
            inputs += input_entry + ","

        outputs = ""
        for ind in output_inds:
            if ind not in ind_to_char:
                raise ValueError(
                    f"Output index '{ind}' not found in input tensors. "
                    f"Available: {set(ind_to_char.keys())}"
                )
            outputs += ind_to_char[ind]

        # Also store char-based size dict for cotengra compatibility
        self._size_dict = {ind_to_char[ind]: size for ind, size in ind_sizes.items()}
        self._einsum_expr = inputs[:-1] + "->" + outputs

    @property
    def einsum_expr(self) -> str:
        """The einsum expression string."""
        return self._einsum_expr

    @property
    def size_dict(self) -> dict[str, int]:
        """Mapping of single-char indices (used in einsum_expr) to their sizes.

        This is for cotengra compatibility where einsum expressions use
        single-character indices.
        """
        return self._size_dict

    @property
    def quantum_tensors(self) -> list[QuantumTensor]:
        """Quantum circuit tensors."""
        return self._qtensors

    @property
    def classical_tensors(self) -> list[CTensor]:
        """Classical tensors."""
        return self._ctensors

    @property
    def input_tensors(self) -> list[TensorSpec]:
        """Input tensor specifications."""
        return self._input_tensors

    @property
    def output_inds(self) -> tuple[str, ...]:
        """Output indices."""
        return self._output_inds

    @staticmethod
    def from_circuit(circuit: QuantumCircuit) -> HEinsum:
        """Create a HEinsum specification from a quantum circuit.

        This method decomposes the circuit into quantum tensors using
        the quantum-pseudo-density (QPD) representation.

        Args:
            circuit: The quantum circuit to convert.

        Returns:
            HEinsum specification representing the circuit.
        """
        return HEinsum(
            qtensors=[QuantumTensor(circuit)],
            ctensors=[],
            input_tensors=[],
            output_inds=(),
        )

    def to_dummy_tn(
        self, seed: int | None = None
    ) -> tuple[ctg.ContractionTree | None, list[np.ndarray]]:
        """Create dummy random arrays and contraction tree for benchmarking.

        This creates random numpy arrays matching the shapes of the HEinsum
        tensors. Useful for measuring classical contraction time without
        running quantum circuits.

        Args:
            optimize: Contraction path optimizer (default: "auto-hq").
            seed: Random seed for reproducibility.

        Returns:
            A tuple of (arrays, tree) where arrays is a list of random numpy
            arrays matching the tensor shapes, and tree is the optimized
            contraction tree from cotengra.
        """

        if seed is not None:
            np.random.seed(seed)

        arrays = []
        inputs = []

        # Collect all tensors (quantum, classical, input)
        all_tensors = (
            list(self._qtensors) + list(self._ctensors) + list(self._input_tensors)
        )

        for tensor in all_tensors:
            if tensor.shape:
                data = np.random.randn(*tensor.shape).astype(np.float64)
            else:
                # Scalar tensor (empty shape)
                data = np.array(np.random.randn(), dtype=np.float64)
            arrays.append(data)

        if len(all_tensors) == 1:
            return None, arrays

        # Get optimized contraction tree (parallel=False to avoid semaphore leaks)
        opt = ctg.HyperOptimizer(parallel=False, progbar=False)
        inputs, outputs = ctg.utils.eq_to_inputs_output(self.einsum_expr)
        tree = opt.search(inputs, outputs, self.size_dict)

        return tree, arrays


# =============================


def rand_regular_heinsum(
    n_quantum: int,
    n_classical: int,
    reg: int = 3,
    q_bond_dim: int = 2,
    c_bond_dim: int = 8,
    seed: int | None = None,
) -> HEinsum:
    """Generate a random n-regular hybrid tensor network.

    Creates an n-regular random graph where each tensor has exactly `reg` bonds.
    Bonds between quantum tensors use small dimensions (q_bond_dim), while
    bonds involving only classical tensors use larger dimensions (c_bond_dim).
    This allows scaling classical contraction complexity without affecting
    quantum tensor sizes.

    Args:
        n_quantum: Number of quantum tensors.
        n_classical: Number of classical tensors.
        reg: Number of bonds per tensor (regularity). Must satisfy
            n_total * reg being even.
        q_bond_dim: Bond dimension for edges touching quantum tensors.
        c_bond_dim: Bond dimension for edges between classical tensors only.
        seed: Random seed for reproducibility.

    Returns:
        HEinsum: A random n-regular hybrid tensor network.

    Example:
        >>> # 5 quantum + 10 classical tensors, 4-regular, small q-bonds, large c-bonds
        >>> h = rand_regular_heinsum(5, 10, reg=4, q_bond_dim=2, c_bond_dim=32)
        >>> # Quantum tensors stay small: 2^4 = 16 elements max
        >>> # Classical contraction cost scales with c_bond_dim
    """
    if seed is not None:
        np.random.seed(seed)

    n_total = n_quantum + n_classical

    if (n_total * reg) % 2 != 0:
        raise ValueError(
            f"n_total * reg must be even for regular graph. "
            f"Got {n_total} * {reg} = {n_total * reg}"
        )

    # Generate random regular graph using configuration model
    # Each tensor has `reg` "half-edges" (stubs)
    stubs = []
    for t in range(n_total):
        stubs.extend([t] * reg)

    # Shuffle and pair up stubs to form edges
    np.random.shuffle(stubs)
    edges = []
    for i in range(0, len(stubs), 2):
        t1, t2 = stubs[i], stubs[i + 1]
        edges.append((min(t1, t2), max(t1, t2)))

    # Remove self-loops and multi-edges by resampling if needed
    # (Simple rejection: just keep valid edges)
    edge_set = set()
    valid_edges = []
    for t1, t2 in edges:
        if t1 != t2 and (t1, t2) not in edge_set:
            edge_set.add((t1, t2))
            valid_edges.append((t1, t2))

    # Build tensor indices from edges
    tensor_inds: list[list[tuple[str, int]]] = [[] for _ in range(n_total)]

    for idx, (t1, t2) in enumerate(valid_edges):
        ind_name = f"e{idx}"

        # Determine bond dimension based on whether quantum tensors are involved
        if t1 < n_quantum or t2 < n_quantum:
            # At least one quantum tensor - use small bond dim
            dim = q_bond_dim
        else:
            # Both classical - use large bond dim
            dim = c_bond_dim

        tensor_inds[t1].append((ind_name, dim))
        tensor_inds[t2].append((ind_name, dim))

    # Create quantum tensors
    qtensors = []
    for t in range(n_quantum):
        inds = tuple(name for name, _ in tensor_inds[t])
        shape = tuple(size for _, size in tensor_inds[t])

        if not shape:
            # Tensor with no edges - give it a trivial shape
            inds = (f"trivial_q{t}",)
            shape = (1,)

        qtensor = QuantumTensor.from_shape(shape, inds)
        qtensors.append(qtensor)

    # Create classical tensors
    ctensors = []
    for t in range(n_quantum, n_total):
        inds = tuple(name for name, _ in tensor_inds[t])
        shape = tuple(size for _, size in tensor_inds[t])

        if not shape:
            # Tensor with no edges - give it a trivial shape
            inds = (f"trivial_c{t}",)
            shape = (1,)
            data = np.array([1.0], dtype=np.float64)
        else:
            data = np.random.randn(*shape).astype(np.float64)

        ctensors.append(CTensor(data, inds))

    return HEinsum(
        qtensors=qtensors,
        ctensors=ctensors,
        input_tensors=[],
        output_inds=(),
    )
