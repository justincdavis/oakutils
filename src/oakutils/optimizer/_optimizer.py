from __future__ import annotations

import time
import itertools
from typing import Callable, Any
from collections import deque

import depthai as dai
from typing_extensions import Self

from ._grid_search import grid_search
    

class Optimizer:
    """Class for optimizing a pipeline onboard an OAK camera."""

    def __init__(
        self: Self,
        algorithm: str = "grid",
        max_measure_time: float = 10.0,
        measure_trials: int = 1,
    ) -> None:
        """Use to create an instance of the class.

        Parameters
        ----------
        algorithm : str, optional
            The algorithm to use for optimization, by default "grid"
        max_measure_time : float, optional
            The amount of time to measure the pipeline for, by default 10.0
        measure_trials : int, optional
            The number of times to measure the pipeline, by default 1

        Raises
        ------
        ValueError
            If the algorithm is invalid
        """
        if algorithm not in ["grid"]:
            raise ValueError(f"Invalid algorithm {algorithm}")
        if algorithm == "grid":
            self._algorithm = grid_search
        self._max_measure_time = max_measure_time
        self._measure_trials = measure_trials

    def measure(
        self: Self,
        pipeline_func: Callable[
            [dai.Pipeline, dict[str, Any]], list[Callable[[dai.Device], None]]
        ],
        pipeline_args: dict[str, Any],
    ) -> tuple[float, float, dict[str, float]]:
        """Use to measure the FPS of a pipeline.

        Parameters
        ----------
        pipeline_func : Callable[[dai.Pipeline, dict[str, Any]], list[Callable[[dai.Device], None]]]
            The function to generate a pipeline
        pipeline_args : dict[str, Any]
            The arguments to measure

        Notes
        -----
        This function will set XLinkChunkSize to 0 for the pipeline.

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
        # overall averaging
        fps: list[float] = []
        latencies: list[float] = []
        all_latencies: list[dict[str, float]] = []

        # run each measure trial
        for trial in range(self._measure_trials):
            # print(f"Running trial {trial + 1} of {self._measure_trials}")
            # create the pipeline
            pipeline: dai.Pipeline = dai.Pipeline()
            device_funcs = pipeline_func(pipeline, pipeline_args)
            pipeline.setXLinkChunkSize(0)

            # create measurement variables
            stopped = False
            data_times: dict[str, list[float]] = {}
            cycle_times: list[float] = []

            # create the device & measure
            with dai.Device(pipeline) as device:
                # run any functions to start the pipeline
                for func in device_funcs:
                    func(device)
                # gather queues and allocate lists
                queue_names: list[str] = device.getOutputQueueNames()
                for queue in queue_names:
                    data_times[queue] = []
                queues: dict = {q: device.getOutputQueue(q) for q in queue_names}

                # run pipelines
                past_length = 10
                past = deque(maxlen=past_length)
                t0 = time.perf_counter()
                while not stopped:
                    t1 = time.perf_counter()
                    for queue_name, queue in queues.items():
                        data = queue.get()
                        data_times[queue_name].append(
                            (dai.Clock.now() - data.getTimestamp()).total_seconds()
                            * 1000
                        )
                    t2 = time.perf_counter()
                    cycle_time = t2 - t1
                    cycle_times.append(cycle_time)
                    past.append(cycle_time)
                    elapsed = t2 - t0
                    # print(f"   Elapsed: {elapsed:.2f}s")
                    stopped1 = elapsed > self._max_measure_time
                    stopped2 = (sum(past) / past_length) == cycle_time
                    stopped = stopped1 or stopped2

            # compute average latency
            avg_latencies: dict[str, float] = {}
            for queue_name, data in data_times.items():
                avg_latencies[queue_name] = sum(data) / len(data)
            avg_latency = sum(avg_latencies.values()) / len(avg_latencies)

            # compute average cycle time
            avg_cycle_time = sum(cycle_times) / len(cycle_times)

            fps.append(1000.0 / avg_cycle_time)
            latencies.append(avg_latency)
            all_latencies.append(avg_latencies)

        # compute average fps
        avg_fps = sum(fps) / len(fps)
        avg_latencies = sum(latencies) / len(latencies)
        avg_all_latencies: dict[str, float] = {}
        for key in all_latencies[0].keys():
            data = [latency[key] for latency in all_latencies]
            avg_all_latencies[key] = sum(data) / len(data)

        return avg_fps, avg_latencies, avg_all_latencies

    def optimize(
        self: Self,
        pipeline_func: Callable[
            [dai.Pipeline, dict[str, Any]], list[Callable[[dai.Device], None]]
        ],
        pipeline_args: dict[str, list[Any]],
        objective_func: Callable[
            [list[tuple[tuple[float, float, dict[str, float]], dict[str, Any]]]],
            dict[str, Any],
        ],
    ) -> dict[str, Any]:
        """Use to generate optimized arguments for a pipeline.

        Parameters
        ----------
        pipeline_func : Callable[[dai.Pipeline, dict[str, Any]], list[Callable[[dai.Device], None]]]
            The function to generate a pipeline
        pipeline_args : dict[str, list[Any]]
            The arguments to optimize
        objective_func : Callable[[list[tuple[tuple[float, float, dict[str, float]], dict[str, Any]]]], dict[str, Any]]
            The function to use to choose the best arguments

        Returns
        -------
        dict[str, Any]
            The optimized arguments
        """
        possible_args: list[dict[str, Any]] = [
            dict(zip(pipeline_args.keys(), values))
            for values in itertools.product(*pipeline_args.values())
        ]
        results = grid_search(pipeline_func, possible_args, self.measure)

        return objective_func(results)
