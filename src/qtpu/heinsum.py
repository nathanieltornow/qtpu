"""Hybrid einsum API for specifying tensor network contractions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cotengra as ctg
import numpy as np

from qtpu.tensor import QuantumTensor, TensorSpec, CTensor

if TYPE_CHECKING:
    from qtpu.evaluators._evaluator import CircuitTensorEvaluator
    from qtpu.runtime import CompiledHybridKernel


class HEinsum:
    """High-level API for specifying hybrid tensor network contractions.

    Allows users to specify quantum tensors, classical tensors, and input specs,
    then automatically generates optimized contraction trees.

    Example:
        >>> qtensor1 = QuantumTensor(circuit1)
        >>> qtensor2 = QuantumTensor(circuit2)
        >>> ctensor = CTensor(data, inds=("k", "l"))
        >>> input_spec = TensorSpec(shape=(4, 5), inds=("l", "m"))
        >>>
        >>> heinsum = HEinsum(
        ...     qtensors=[qtensor1, qtensor2],
        ...     ctensors=[ctensor],
        ...     input_tensors=[input_spec],
        ...     output_inds=("i", "m"),
        ... )
        >>>
    """

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

        self._size_dict = ind_sizes
        # Also store char-based size dict for cotengra compatibility
        self._char_size_dict = {
            ind_to_char[ind]: size for ind, size in ind_sizes.items()
        }
        self._einsum_expr = inputs[:-1] + "->" + outputs

    @property
    def einsum_expr(self) -> str:
        """The einsum expression string."""
        return self._einsum_expr

    @property
    def size_dict(self) -> dict[str, int]:
        """Mapping of index names to their sizes."""
        return self._size_dict

    @property
    def char_size_dict(self) -> dict[str, int]:
        """Mapping of single-char indices (used in einsum_expr) to their sizes.

        This is for cotengra compatibility where einsum expressions use
        single-character indices.
        """
        return self._char_size_dict

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
