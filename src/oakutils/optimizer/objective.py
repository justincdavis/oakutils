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
Module containing objective functions for choosing the best set of pipeline arguments.

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

from typing import Any


def highest_fps(
    options: list[tuple[tuple[float, float, dict[str, float]], dict[str, Any]]],
) -> tuple[dict[str, Any], tuple[float, float, dict[str, float]]]:
    """
    Use to get the set of arguments with the highest fps.

    Parameters
    ----------
    options : list[tuple[tuple[float, float, dict[str, float]], dict[str, Any]]]
        The list of options to choose from. A list of tuples, where the first
        element is the result of `Optimizer.measure` and the second element is the
        arguments used to generate the pipeline.

    Returns
    -------
    dict[str, Any]
        The arguments with the highest fps
    tuple[float, float, dict[str, float]]
        The measurement results
    """
    fps_values = [option[0][0] for option in options]
    max_fps = max(fps_values)
    max_fps_index = fps_values.index(max_fps)
    return options[max_fps_index][1], options[max_fps_index][0]


def lowest_avg_latency(
    options: list[tuple[tuple[float, float, dict[str, float]], dict[str, Any]]],
) -> tuple[dict[str, Any], tuple[float, float, dict[str, float]]]:
    """
    Use to get the set of arguments with the lowest avg latency.

    Parameters
    ----------
    options : list[tuple[tuple[float, float, dict[str, float]], dict[str, Any]]]
        The list of options to choose from. A list of tuples, where the first
        element is the result of `Optimizer.measure` and the second element is the
        arguments used to generate the pipeline.

    Returns
    -------
    dict[str, Any]
        The arguments with the lowest avg latency
    tuple[float, float, dict[str, float]]
        The measurement results
    """
    avg_latency_values = [option[0][1] for option in options]
    min_avg_latency = min(avg_latency_values)
    min_avg_latency_index = avg_latency_values.index(min_avg_latency)
    return options[min_avg_latency_index][1], options[min_avg_latency_index][0]


def lowest_latency(
    options: list[tuple[tuple[float, float, dict[str, float]], dict[str, Any]]],
    stream: str,
) -> tuple[dict[str, Any], tuple[float, float, dict[str, float]]]:
    """
    Use to get the set of arguments with the lowest latency for a specific stream.

    Parameters
    ----------
    options : list[tuple[tuple[float, float, dict[str, float]], dict[str, Any]]]
        The list of options to choose from. A list of tuples, where the first
        element is the result of `Optimizer.measure` and the second element is the
        arguments used to generate the pipeline.
    stream : str
        The name of the stream to get the lowest latency for

    Returns
    -------
    dict[str, Any]
        The arguments with the lowest latency for the specified stream
    tuple[float, float, dict[str, float]]
        The measurement results for the specified stream
    """
    latency_values = [option[0][2][stream] for option in options]
    min_latency = min(latency_values)
    min_latency_index = latency_values.index(min_latency)
    return options[min_latency_index][1], options[min_latency_index][0]
