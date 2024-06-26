# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Module for performing filtering on typical datatypes.

Submodules
----------
wls
    Module for performing WLS filtering.

Classes
-------
WLSFilter
    A class for computing the weighted-least-squares filter,
    on disparity images.
"""

from __future__ import annotations

import logging

from . import wls
from .wls import WLSFilter

_log = logging.getLogger(__name__)

__all__ = [
    "WLSFilter",
    "wls",
]

_log.debug("Loaded filters")
