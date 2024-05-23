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
"""
Submodule for making sending and receiving data from the OAK-D easier.

Classes
-------
Buffer
    Class for creating a buffer for sending and receiving data from the OAK-D.
MultiBuffer
    Class for creating a buffer for sending and receiving multiple data streams from the OAK-D.
SimpleBuffer
    Class for creating a buffer for sending and receiving data from the OAK-D.

Functions
---------
create_synced_buffer
    Creates a function for getting packets of data from multiple streams.

"""
from __future__ import annotations

from ._buffer import Buffer
from ._funcs import create_synced_buffer
from ._multi_buffer import MultiBuffer
from ._simple_buffer import SimpleBuffer

__all__ = ["Buffer", "MultiBuffer", "SimpleBuffer", "create_synced_buffer"]
