# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Module for converting between different data types in the blob definitions.

Functions
---------
convert_to_fp16
    Use to convert a Uint8 tensor with double columns to a float16 tensor with single columns.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

_log = logging.getLogger(__name__)


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
    _log.warning("Conversion of fp16 to uint8 is no longer needed as of depthai 2.22+")
    fp16: torch.Tensor = 256.0 * tensor[:, :, :, 1::2] + tensor[:, :, :, ::2]
    return fp16
