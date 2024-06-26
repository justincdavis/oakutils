# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
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
    input_queue_size: int | None = None,
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
        The queue size of the input, by default None
    input_reuse : Optional[bool], optional
        Whether to reuse the previous message, by default None
    input_blocking : Optional[bool], optional
        Whether to block the input, by default None
    input_wait_for_message : Optional[bool], optional
        Whether to wait for a message, by default None

    Returns
    -------
    dai.node.XLinkOut
        The XLinkOut node

    """
    xout = pipeline.createXLinkOut()
    xout.setStreamName(stream_name)
    input_link.link(xout.input)

    if input_queue_size is not None:
        xout.input.setQueueSize(input_queue_size)
    if input_reuse is not None:
        xout.input.setReusePreviousMessage(input_reuse)
    if input_blocking is not None:
        xout.input.setBlocking(input_blocking)
    if input_wait_for_message is not None:
        xout.input.setWaitForMessage(input_wait_for_message)

    return xout
