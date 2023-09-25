from __future__ import annotations

import time
import itertools
from typing import TYPE_CHECKING, Callable, Any

import depthai as dai

if TYPE_CHECKING:
    from typing_extensions import Self


class Optimizer:
    """Class for optimizing a pipeline onboard an OAK camera."""

    def __init__(self: Self, measure_time: float = 10.0) -> None:
        """Use to create an instance of the class."""
        self._measure_time = measure_time
        self._fps: dict[dict[str, list[Any]], float] = {}

    def _measure(self: Self, pipeline_func: Callable, pipeline_args: dict[str, Any]) -> tuple[float, float, dict[str, float]]:
        """Use to measure the FPS of a pipeline.
        
        Parameters
        ----------
        pipeline_func : Callable
            The function to generate a pipeline
        pipeline_args : dict[str, Any]
            The arguments to measure
        
        Returns
        -------
        float
            The average cycle time of the whole pipeline (FPS)
        float
            The average latency of all data packets in the pipeline
        dict[str, float]
            The average latency of each data packet in the pipeline
            Key is the name of the data stream
        """
        # create the pipeline
        pipeline: dai.Pipeline = dai.Pipeline()
        pipeline_func(pipeline, pipeline_args)

        # create measurement variables
        stopped = False
        data_times: dict[str, list[float]] = {}
        cycle_times: list[float] = []

        # create the device & measure
        with dai.Device(pipeline) as device:
            # gather queues and allocate lists
            queue_names: list[str] = device.getOutputQueueNames()
            for queue in queue_names:
                data_times[queue] = []
            queues = [device.getOutputQueue(q) for q in queue_names]

            # run pipelines
            while not stopped:
                t0 = time.perf_counter()
                for queue in queues:
                    data = queue.get()
                    data_times[queue].append((dai.Clock.now() - data.getTimestamp()).total_seconds() * 1000)
                elapsed = time.perf_counter() - t0
                cycle_times.append(elapsed)
                stopped = elapsed > self._measure_time
        
        # compute average latency
        avg_latencies: dict[str, float] = {}
        for queue_name, data in data_times.items():
            avg_latencies[queue_name] = sum(data) / len(data)
        
        # compute average cycle time
        avg_cycle_time = sum(cycle_times) / len(cycle_times)

    def optimize(self: Self, pipeline_func: Callable[[dai.Pipeline, dict[str, Any]], None], pipeline_args: dict[str, list[Any]]) -> dict[str, Any]:
        """Use to generate optimized arguments for a pipeline.
        
        Parameters
        ----------
        pipeline_func : Callable[[dai.Pipeline, dict[str, Any]], None]
            The function to generate a pipeline
        pipeline_args : dict[str, list[Any]]
            The arguments to optimize
        
        Returns
        -------
        dict[str, Any]
            The optimized arguments
        """
        possible_args: list[dict[str, Any]] = [
            dict(zip(pipeline_args.keys(), values))
            for values in itertools.product(*pipeline_args.values())
        ]
        for args in possible_args:
            self._fps[args] = self._measure(pipeline_func, args)

