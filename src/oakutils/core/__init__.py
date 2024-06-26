# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Submodule for core utilities for working with the OAK-D.

Functions
---------
create_device
    Create a DepthAI device object from a pipeline.

"""

from __future__ import annotations

import logging

from ._device import create_device

_log = logging.getLogger(__name__)

__all__ = ["create_device"]

_log.debug("Loaded core")
