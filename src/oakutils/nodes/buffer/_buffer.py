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

import numpy as np

from ._multi_buffer import MultiBuffer

if TYPE_CHECKING:
    import depthai as dai
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
        self._buffer = MultiBuffer(device, input_streams, output_streams)

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
