# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
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
