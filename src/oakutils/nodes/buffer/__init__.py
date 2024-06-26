# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
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

import logging

from ._buffer import Buffer
from ._funcs import create_synced_buffer
from ._multi_buffer import MultiBuffer
from ._simple_buffer import SimpleBuffer

_log = logging.getLogger(__name__)

__all__ = ["Buffer", "MultiBuffer", "SimpleBuffer", "create_synced_buffer"]

_log.debug("Loaded nodes.buffer")
