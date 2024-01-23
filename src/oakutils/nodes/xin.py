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
Module for creating and using depthai Xin nodes.

Functions
---------
create_xin
    Use to create an xin node
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import depthai as dai


def create_xin(
    pipeline: dai.Pipeline,
    stream_name: str,
    max_data_size: int = 6144000,
) -> dai.node.XLinkIn:
    """
    Use to create an XLinkIn node.

    Parameters
    ----------
    pipeline : dai.Pipeline
        The pipeline to add the XLinkIn node to
    stream_name : str
        The name of the stream
    max_data_size : int, optional
        The maximum data size, by default 6144000 bytes

    Returns
    -------
    dai.node.XLinkIn
        The XLinkIn node
    """
    xin = pipeline.createXLinkIn()
    xin.setStreamName(stream_name)
    xin.setMaxDataSize(max_data_size)

    return xin
