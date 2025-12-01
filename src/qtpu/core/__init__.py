"""Core tensor types for QTPU.

This module provides the fundamental tensor types used throughout QTPU:

- ISwitch: Instruction for parameterized quantum circuit selection
- QuantumTensor: Tensor of quantum circuits indexed by ISwitch parameters
- CTensor: Classical tensor with named indices
- TensorSpec: Lightweight specification of tensor shape and indices
- HEinsum: Hybrid einsum specification for tensor network contractions
"""

from dataclasses import dataclass

from qtpu.core.iswitch import ISwitch
from qtpu.core.qtensor import QuantumTensor
from qtpu.core.ctensor import CTensor
from qtpu.core.heinsum import HEinsum, rand_regular_heinsum


@dataclass
class TensorSpec:
    """Lightweight specification of a tensor's shape and indices.
    
    Used for specifying input tensors to HEinsum without providing data.
    
    Attributes:
        shape: The dimensions of the tensor.
        inds: Names of the tensor indices (quimb-style).
    """
    shape: tuple[int, ...]
    inds: tuple[str, ...]


__all__ = [
    "ISwitch",
    "QuantumTensor",
    "CTensor",
    "TensorSpec",
    "HEinsum",
    "rand_regular_heinsum",
]
