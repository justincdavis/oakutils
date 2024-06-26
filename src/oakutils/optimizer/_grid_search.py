# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import depthai as dai


def grid_search(
    pipeline_func: Callable[
        [dai.Pipeline, dict[str, Any]],
        list[Callable[[dai.DeviceBase], None]],
    ],
    possible_args: list[dict[str, Any]],
    measure_func: Callable[
        [
            Callable[
                [dai.Pipeline, dict[str, Any]],
                list[Callable[[dai.DeviceBase], None]],
            ],
            dict[str, Any],
        ],
        tuple[float, float, dict[str, float]],
    ],
    objective_func: Callable[
        [list[tuple[tuple[float, float, dict[str, float]], dict[str, Any]]]],
        tuple[dict[str, Any], tuple[float, float, dict[str, float]]],
    ],
) -> tuple[dict[str, Any], tuple[float, float, dict[str, float]]]:
    """
    Use to run a grid search and find all possible measurements.

    Parameters
    ----------
    pipeline_func : Callable[[dai.Pipeline, dict[str, Any]], list[Callable[[dai.DeviceBase], None]]]
        The function to generate a pipeline
    possible_args : list[dict[str, Any]]
        The arguments to measure
    measure_func : Callable[[Callable[[dai.Pipeline, dict[str, Any]], list[Callable[[dai.DeviceBase], None]]], dict[str, Any]], tuple[float, float, dict[str, float]]]
        The function to measure the pipeline
    objective_func : Callable[[list[tuple[tuple[float, float, dict[str, float]], dict[str, Any]]]], dict[str, Any]]
        The function to use to choose the best arguments

    Returns
    -------
    dict[str, Any]
        The arguments maximizing the objective functions
    tuple[float, float, dict[str, float]]
        The best measurement results

    """
    results: list[tuple[tuple[float, float, dict[str, float]], dict[str, Any]]] = []
    for arg in possible_args:
        result = measure_func(pipeline_func, arg)
        results.append((result, arg))
    return objective_func(results)
