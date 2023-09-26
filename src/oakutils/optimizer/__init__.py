"""Module for optimizing pipelines.

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
from . import objective
from .objective import highest_fps, lowest_avg_latency, lowest_latency
from ._optimizer import Optimizer

__all__ = [
    "objective",
    "highest_fps",
    "lowest_avg_latency",
    "lowest_latency",
    "Optimizer",
]
