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
