# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from ._buffer import Buffer

if TYPE_CHECKING:
    import depthai as dai
    import numpy as np
    from typing_extensions import Self


class MultiBuffer:
    """Class for creating multiple Buffers for sending and receiving data from the OAK-D."""

    def __init__(
        self: Self,
        device: dai.DeviceBase,
        input_streams: Sequence[str | list[str]],
        output_streams: Sequence[str | list[str]],
    ) -> None:
        """
        Create the multi buffer from streams.

        Parameters
        ----------
        device : dai.DeviceBase
            The OAK-D device which the streams are built on.
        input_streams : list[str | list[str]]
            The input stream names for the buffer to send data through.
        output_streams : list[str | list[str]]
            The output stream names for the buffer to receive data from.

        """
        self._buffers: list[Buffer] = []
        for input_stream, output_stream in zip(input_streams, output_streams):
            self._buffers.append(Buffer(device, input_stream, output_stream))

    def __call__(
        self: Self,
        data: list[np.ndarray | list[np.ndarray]],
    ) -> list[dai.ADatatype | list[dai.ADatatype]]:
        """
        Cycle data through the multi buffer.

        Parameters
        ----------
        data : list[np.ndarray | list[np.ndarray]]
            The data to cycle through the buffer.

        Returns
        -------
        list[dai.ADatatype | list[dai.ADatatype]]
            The data received from the buffer.

        """
        return self.cycle(data)

    def cycle(
        self: Self,
        data: list[np.ndarray | list[np.ndarray]],
    ) -> list[dai.ADatatype | list[dai.ADatatype]]:
        """
        Cycle data through the multi buffer.

        Parameters
        ----------
        data : list[np.ndarray | list[np.ndarray]]
            The data to cycle through the buffer.

        Returns
        -------
        list[dai.ADatatype | list[dai.ADatatype]]
            The data received from the buffer.

        """
        return [buffer.cycle(d) for buffer, d in zip(self._buffers, data)]

    def send(
        self: Self,
        data: list[np.ndarray | list[np.ndarray]],
    ) -> None:
        """
        Send data through the multi buffer.

        Parameters
        ----------
        data : list[np.ndarray | list[np.ndarray]]
            The data to send through the buffer.

        """
        for buffer, d in zip(self._buffers, data):
            buffer.send(d)

    def receive(
        self: Self,
    ) -> list[dai.ADatatype | list[dai.ADatatype]]:
        """
        Receive data from the multi buffer.

        Returns
        -------
        list[dai.ADatatype | list[dai.ADatatype]]
            The data received from the buffer.

        """
        return [buffer.receive() for buffer in self._buffers]
