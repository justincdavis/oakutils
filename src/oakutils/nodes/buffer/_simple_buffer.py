# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

import depthai as dai

if TYPE_CHECKING:
    import numpy as np
    from typing_extensions import Self


class SimpleBuffer:
    """Buffer for sending and receiving data from OAK-D."""

    def __init__(
        self: Self,
        device: dai.DeviceBase,
        input_stream: str,
        output_stream: str,
    ) -> None:
        """
        Create the buffer.

        Parameters
        ----------
        device : dai.DeviceBase
            The OAK-D device which the streams are built on.
        input_stream : str | list[str]
            The input stream name or names for the buffer to send data through.
        output_stream : str | list[str]
            The output stream name or names for the buffer to receive data from.

        """
        self._buffer = dai.Buffer()
        self._input_stream = input_stream
        self._output_stream = output_stream
        self._input_queue: dai.DataInputQueue = device.getInputQueue(self._input_stream)  # type: ignore[attr-defined]
        self._output_queue: dai.DataOutputQueue = device.getOutputQueue(  # type: ignore[attr-defined]
            self._output_stream,
        )

    def __call__(
        self: Self,
        data: np.ndarray,
    ) -> dai.ADatatype:
        """
        Cycle data through the buffer.

        Parameters
        ----------
        data : np.ndarray
            The data to cycle through the buffer.

        Returns
        -------
        dai.ADatatype
            The data cycled through the buffer.

        """
        return self.cycle(data)

    def cycle(
        self: Self,
        data: np.ndarray,
    ) -> dai.ADatatype:
        """
        Cycle data through the buffer.

        Parameters
        ----------
        data : np.ndarray
            The data to cycle through the buffer.

        Returns
        -------
        dai.ADatatype
            The data cycled through the buffer.

        """
        self.send(data)
        return self.receive()

    def send(self: Self, data: np.ndarray) -> None:
        """
        Send data through the buffer.

        Parameters
        ----------
        data : np.ndarray
            The data to send through the buffer.

        """
        self._buffer.setData(data)
        self._input_queue.send(self._buffer)

    def receive(self: Self) -> dai.ADatatype:
        """
        Receive data from the buffer.

        Returns
        -------
        dai.ADataType
            The data received from the buffer.

        """
        return self._output_queue.get()
