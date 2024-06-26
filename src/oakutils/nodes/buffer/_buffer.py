# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

import depthai as dai
import numpy as np

if TYPE_CHECKING:
    from typing_extensions import Self


class Buffer:
    """Buffer for sending and receiving data from OAK-D."""

    def __init__(
        self: Self,
        device: dai.DeviceBase,
        input_stream: str | list[str],
        output_stream: str | list[str],
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
        if isinstance(input_stream, str):
            input_streams = [input_stream]
        else:
            input_streams = input_stream
        if isinstance(output_stream, str):
            output_streams = [output_stream]
        else:
            output_streams = output_stream
        self._buffer = _Buffer(device, input_streams, output_streams)

    def __call__(
        self: Self,
        data: np.ndarray | list[np.ndarray],
    ) -> dai.ADatatype | list[dai.ADatatype]:
        """
        Cycle data through the buffer.

        Parameters
        ----------
        data : np.ndarray | list[np.ndarray]
            The data to cycle through the buffer.

        Returns
        -------
        dai.ADatatype | list[dai.ADatatype]
            The data cycled through the buffer.

        """
        return self.cycle(data)

    def cycle(
        self: Self,
        data: np.ndarray | list[np.ndarray],
    ) -> dai.ADatatype | list[dai.ADatatype]:
        """
        Cycle data through the buffer.

        Parameters
        ----------
        data : np.ndarray | list[np.ndarray]
            The data to cycle through the buffer.

        Returns
        -------
        dai.ADatatype | list[dai.ADatatype]
            The data cycled through the buffer.

        """
        self.send(data)
        return self.receive()

    def send(self: Self, data: np.ndarray | list[np.ndarray]) -> None:
        """
        Send data through the buffer.

        Parameters
        ----------
        data : np.ndarray | list[np.ndarray]
            The data to send through the buffer.

        """
        datas = [data] if isinstance(data, np.ndarray) else data
        self._buffer.send(datas)

    def receive(self: Self) -> dai.ADatatype | list[dai.ADatatype]:
        """
        Receive data from the buffer.

        Returns
        -------
        dai.ADataType | list[dai.ADatatype]
            The data received from the buffer.

        """
        data = self._buffer.receive()
        if len(data) == 1:
            return data[0]
        return data


class _Buffer:
    """Buffer for sending and receiving data from OAK-D."""

    def __init__(
        self: Self,
        device: dai.DeviceBase,
        input_streams: list[str],
        output_streams: list[str],
    ) -> None:
        """
        Create the buffer.

        Parameters
        ----------
        device : dai.DeviceBase
            The OAK-D device which the streams are built on.
        input_streams : list[str]
            A list of input stream names for the buffer to send data through,
            when multiple inputs from host are needed.
        output_streams : list[str]
            The output stream names for the buffer to receive data from,
            use multiple when want batches of data.

        """
        self._buffers: list[dai.Buffer] = []
        self._input_queues: list[dai.DataInputQueue] = []
        self._input_streams = input_streams
        for stream in self._input_streams:
            self._input_queues.append(device.getInputQueue(stream))  # type: ignore[attr-defined]
            self._buffers.append(dai.Buffer())
        self._output_queues: list[dai.DataOutputQueue] = []
        self._output_streams = output_streams
        for stream in self._output_streams:
            self._output_queues.append(device.getOutputQueue(stream))  # type: ignore[attr-defined]

    def __call__(
        self: Self,
        data: list[np.ndarray],
    ) -> list[dai.ADatatype]:
        """
        Cycle data through the buffer.

        Parameters
        ----------
        data : list[np.ndarray]
            The list of data to cycle through the buffer.

        Returns
        -------
        list[dai.ADatatype]
            The data cycled through the buffer.

        """
        return self.cycle(data)

    def cycle(
        self: Self,
        data: list[np.ndarray],
    ) -> list[dai.ADatatype]:
        """
        Cycle data through the buffer.

        Parameters
        ----------
        data : list[np.ndarray]
            The list of data to cycle through the buffer.

        Returns
        -------
        list[dai.ADatatype]
            The data cycled through the buffer.

        """
        self.send(data)
        return self.receive()

    def send(self: Self, data: list[np.ndarray]) -> None:
        """
        Send data through the buffer.

        Parameters
        ----------
        data : list[np.ndarray]
            The list of data to send through the buffer.

        """
        for idx, d in enumerate(data):
            self._buffers[idx].setData(d)
            self._input_queues[idx].send(self._buffers[idx])

    def receive(self: Self) -> list[dai.ADatatype]:
        """
        Receive data from the buffer.

        Returns
        -------
        list[dai.ADatatype]
            The data received from the buffer.

        """
        return [queue.get() for queue in self._output_queues]
