from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def convert_to_fp16(tensor: torch.Tensor) -> torch.Tensor:
    """Converts a Uint8 tensor with double columns to a float16 tensor with single columns.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to convert

    Returns
    -------
    torch.Tensor
        The converted tensor
    """
    return 256.0 * tensor[:, :, :, 1::2] + tensor[:, :, :, ::2]
