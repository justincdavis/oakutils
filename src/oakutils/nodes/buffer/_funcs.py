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

from functools import partial
from typing import TYPE_CHECKING, Callable

from ._buffer import Buffer

if TYPE_CHECKING:
    import depthai as dai


def create_synced_buffer(
    device: dai.DeviceBase,
    streams: list[str],
) -> Callable[[], list[dai.ADatatype]]:
    def _get_packet(buffer: Buffer) -> list[dai.ADatatype]:
        data = buffer.receive()
        if isinstance(data, list):
            return data
        return [data]

    buffer = Buffer(device, [], streams)

    return partial(_get_packet, buffer)
