"""
Module for converting between different data types in the blob definitions.

Functions
---------
convert_to_fp16
    Use to convert a Uint8 tensor with double columns to a float16 tensor with single columns.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def convert_to_fp16(tensor: torch.Tensor) -> torch.Tensor:
    """
    Use to convert a Uint8 tensor with double columns to a float16 tensor with single columns.

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
