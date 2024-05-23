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
from __future__ import annotations

from typing import TYPE_CHECKING

import depthai as dai

if TYPE_CHECKING:
    import numpy as np
    from typing_extensions import Self


class MultiBuffer:
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
