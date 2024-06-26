# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

from ._multi_buffer import MultiBuffer

if TYPE_CHECKING:
    import depthai as dai
    from typing_extensions import Self


class PacketBuffer:
    """Buffer for receiving a packet of outputs from multiple streams."""

    def __init__(
        self: Self,
        device: dai.DeviceBase,
        output_streams: list[str],
    ) -> None:
        """
        Create the buffer.

        Parameters
        ----------
        device : dai.DeviceBase
            The OAK-D device which the streams are built on.
        output_streams : str | list[str]
            The output stream name or names for the buffer to receive data from.

        """
        self._buffer = MultiBuffer(device, [], output_streams)

    def __call__(
        self: Self,
    ) -> list[dai.ADatatype]:
        """
        Get the data from the buffer.

        Returns
        -------
        list[dai.ADatatype]
            The data cycled through the buffer.

        """
        return self.get()

    def get(
        self: Self,
    ) -> list[dai.ADatatype]:
        """
        Receive data from the buffer.

        Returns
        -------
        list[dai.ADatatype]
            The data received from the buffer.

        """
        data = self._buffer.receive()
        new_data: list[dai.ADatatype] = []
        for packet in data:
            # since only list[str] should be given in constructor
            # only dai.ADatatype should be encountered
            # still run the expansion in case, and for type checking
            if isinstance(packet, list):
                new_data.extend(packet)
            else:
                new_data.append(packet)
        return new_data
