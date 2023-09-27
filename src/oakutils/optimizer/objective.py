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
    options: list[tuple[tuple[float, float, dict[str, float]], dict[str, Any]]]
) -> dict[str, Any]:
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
    """
    fps_values = [option[0][0] for option in options]
    max_fps = max(fps_values)
    max_fps_index = fps_values.index(max_fps)
    return options[max_fps_index][1]


def lowest_avg_latency(
    options: list[tuple[tuple[float, float, dict[str, float]], dict[str, Any]]]
) -> dict[str, Any]:
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
    """
    avg_latency_values = [option[0][1] for option in options]
    min_avg_latency = min(avg_latency_values)
    min_avg_latency_index = avg_latency_values.index(min_avg_latency)
    return options[min_avg_latency_index][1]


def lowest_latency(
    stream: str,
    options: list[tuple[tuple[float, float, dict[str, float]], dict[str, Any]]],
) -> dict[str, Any]:
    """
    Use to get the set of arguments with the lowest latency for a specific stream.

    Parameters
    ----------
    stream : str
        The name of the stream to get the lowest latency for
    options : list[tuple[tuple[float, float, dict[str, float]], dict[str, Any]]]
        The list of options to choose from. A list of tuples, where the first
        element is the result of `Optimizer.measure` and the second element is the
        arguments used to generate the pipeline.

    Returns
    -------
    dict[str, Any]
        The arguments with the lowest latency for the specified stream
    """
    latency_values = [option[0][2][stream] for option in options]
    min_latency = min(latency_values)
    min_latency_index = latency_values.index(min_latency)
    return options[min_latency_index][1]
