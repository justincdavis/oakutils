# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Submodule for using the onboard VPU as a standalone processor.

Classes
-------
VPU
    A class for using the onboard VPU as a standalone processor.
"""

from __future__ import annotations

import logging

from ._vpu import VPU

_log = logging.getLogger(__name__)

__all__ = [
    "VPU",
]

_log.debug("Loaded vpu")
