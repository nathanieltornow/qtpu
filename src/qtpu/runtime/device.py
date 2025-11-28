"""Device management for classical tensor operations."""

from __future__ import annotations

from enum import Enum

import torch


class Device(Enum):
    """Computation device for classical tensor operations."""
    
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon


def get_device(device: str | Device | torch.device | None = None) -> torch.device:
    """Get the appropriate torch device.
    
    Args:
        device: Device specification. If None, auto-selects best available.
        
    Returns:
        torch.device for tensor operations.
        
    Example:
        >>> get_device()  # Auto-select (CUDA if available)
        device(type='cuda')
        >>> get_device("cpu")
        device(type='cpu')
        >>> get_device(Device.CUDA)
        device(type='cuda')
    """
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    if isinstance(device, torch.device):
        return device
    if isinstance(device, Device):
        return torch.device(device.value)
    return torch.device(device)
