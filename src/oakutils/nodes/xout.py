from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import depthai as dai


def create_xout(
    pipeline: dai.Pipeline,
    input_link: dai.Node.Output,
    stream_name: str,
) -> dai.node.XLinkOut:
    """Creates an XLinkOut node.

    Parameters
    ----------
    pipeline : dai.Pipeline
        The pipeline to add the XLinkOut node to
    input_link : dai.Node.Output
        The input link to connect to the XLinkOut node
        Example: cam_rgb.preview
    stream_name : str
        The name of the stream

    Returns
    -------
    dai.node.XLinkOut
        The XLinkOut node
    """
    xout = pipeline.createXLinkOut()
    xout.setStreamName(stream_name)
    input_link.link(xout.input)

    return xout
