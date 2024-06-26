# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Module for optimizing pipelines.

Submodules
----------
objective
    Module containing objective functions for choosing the best set of pipeline arguments.

Classes
-------
Optimizer
    Class for optimizing pipelines.

Functions
---------
highest_fps
    Use to get the set of arguments with the highest fps
lowest_avg_latency
    Use to get the set of arguments with the lowest avg latency
lowest_latency
    Use to get the set of arguments with the lowest latency for a specific stream
"""

from __future__ import annotations

import logging

from . import objective
from ._optimizer import Optimizer
from .objective import highest_fps, lowest_avg_latency, lowest_latency

_log = logging.getLogger(__name__)

__all__ = [
    "Optimizer",
    "highest_fps",
    "lowest_avg_latency",
    "lowest_latency",
    "objective",
]

_log.debug("Loaded optimizer")
