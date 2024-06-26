# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

from ._packet_buffer import PacketBuffer

if TYPE_CHECKING:
    import depthai as dai


def create_synced_buffer(
    device: dai.DeviceBase,
    streams: list[str],
) -> PacketBuffer:
    """
    Create a function for getting packets of data from multiple streams.

    Parameters
    ----------
    device : dai.DeviceBase
        The OAK-D device which the streams are built on.
    streams : list[str]
        The output stream names for the buffer to receive data from.

    Returns
    -------
    PacketBuffer
        The buffer for receiving a packet of outputs from multiple streams.

    """
    return PacketBuffer(device, streams)
