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
from . import objective
from ._optimizer import Optimizer
from .objective import highest_fps, lowest_avg_latency, lowest_latency

__all__ = [
    "Optimizer",
    "highest_fps",
    "lowest_avg_latency",
    "lowest_latency",
    "objective",
]
