from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import depthai as dai


def create_xin(
    pipeline: dai.Pipeline,
    stream_name: str,
    max_data_size: int = 6144000,
) -> dai.node.XLinkIn:
    """Creates an XLinkIn node.

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
