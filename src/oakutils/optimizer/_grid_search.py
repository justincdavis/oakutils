from typing import TYPE_CHECKING, Callable, Any

if TYPE_CHECKING:
    import depthai as dai


def grid_search(
    pipeline_func: Callable[
        [dai.Pipeline, dict[str, Any]],
        list[Callable[[dai.Device], None]],
    ],
    possible_args: list[dict[str, Any]],
    measure_func: Callable[
        [
            Callable[
                [dai.Pipeline, dict[str, Any]], list[Callable[[dai.Device], None]]
            ],
            dict[str, Any],
        ],
        tuple[float, float, dict[str, float]],
    ],
) -> list[tuple[tuple[float, float, dict[str, float]], dict[str, Any]]]:
    """Use to run a grid search and find all possible measurements.

    Parameters
    ----------
    pipeline_func : Callable[[dai.Pipeline, dict[str, Any]], list[Callable[[dai.Device], None]]]
        The function to generate a pipeline
    possible_args : list[dict[str, Any]]
        The arguments to measure
    measure_func : Callable[[Callable[[dai.Pipeline, dict[str, Any]], list[Callable[[dai.Device], None]]], dict[str, Any]], tuple[float, float, dict[str, float]]]
        The function to measure the pipeline

    Returns
    -------
    list[tuple[tuple[float, float, dict[str, float]], dict[str, Any]]]
        The list of measurements
    """
    results: list[tuple[tuple[float, float, dict[str, float]], dict[str, Any]]] = []
    for arg in possible_args:
        result = measure_func(pipeline_func, arg)
        results.append((result, arg))
    return results
