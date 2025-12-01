"""Classical tensor representation for hybrid tensor networks."""

from __future__ import annotations

import numpy as np
import torch


class CTensor:
    """A class to represent a classical tensor.

    Parameters
    ----------
    data : torch.Tensor | np.ndarray
        The data of the classical tensor.
    inds : tuple[str, ...]
        The names of the indices in the tensor.

    Attributes:
    ----------
    data : torch.Tensor
        The data of the classical tensor.
    shape : tuple[int, ...]
        The shape of the tensor.
    inds : tuple[str, ...]
        The names of the indices in the tensor.
    """

    def __init__(
        self,
        data: torch.Tensor | np.ndarray,
        inds: tuple[str, ...],
        dtype: torch.dtype = torch.float64,
    ) -> None:
        """Initialize the tensor with given data and indices.

        Args:
            data: The data of the classical tensor (torch.Tensor or np.ndarray).
            inds: The names of the indices in the tensor.
            dtype: The torch dtype for the tensor.
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(dtype)
        elif not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=dtype)
        self._data = data
        self._inds = inds

    @property
    def data(self) -> torch.Tensor:
        """Returns the data of the classical tensor.

        Returns:
            torch.Tensor: The data of the classical tensor.
        """
        return self._data

    @property
    def shape(self) -> tuple[int, ...]:
        """Returns the shape of the tensor.

        Returns:
            tuple[int, ...]: A tuple representing the dimensions of the tensor.
        """
        return tuple(self._data.shape)

    @property
    def inds(self) -> tuple[str, ...]:
        """Returns the indices of the tensor (quimb-style).

        Returns:
            tuple[str, ...]: A tuple of strings representing the indices of the tensor.
        """
        return self._inds
