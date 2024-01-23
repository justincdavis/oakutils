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
Module for creating and using depthai Xout nodes.

Functions
---------
create_xout
    Use to create an xout node
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import depthai as dai


def create_xout(
    pipeline: dai.Pipeline,
    input_link: dai.Node.Output,
    stream_name: str,
    input_queue_size: int = 3,
    *,
    input_reuse: bool | None = None,
    input_blocking: bool | None = None,
    input_wait_for_message: bool | None = None,
) -> dai.node.XLinkOut:
    """
    Use to create an XLinkOut node.

    Parameters
    ----------
    pipeline : dai.Pipeline
        The pipeline to add the XLinkOut node to
    input_link : dai.Node.Output
        The input link to connect to the XLinkOut node
        Example: cam_rgb.preview
    stream_name : str
        The name of the stream
    input_queue_size : int, optional
        The queue size of the input, by default 3
    input_reuse : Optional[bool], optional
        Whether to reuse the previous message, by default None
        If None, will be set to False
    input_blocking : Optional[bool], optional
        Whether to block the input, by default None
        If None, will be set to False
    input_wait_for_message : Optional[bool], optional
        Whether to wait for a message, by default None
        If None, will be set to False

    Returns
    -------
    dai.node.XLinkOut
        The XLinkOut node
    """
    xout = pipeline.createXLinkOut()
    xout.setStreamName(stream_name)
    input_link.link(xout.input)

    if input_reuse is None:
        input_reuse = False
    if input_blocking is None:
        input_blocking = False
    if input_wait_for_message is None:
        input_wait_for_message = False

    xout.input.setQueueSize(input_queue_size)
    xout.input.setReusePreviousMessage(input_reuse)
    xout.input.setBlocking(input_blocking)
    xout.input.setWaitForMessage(input_wait_for_message)

    return xout
