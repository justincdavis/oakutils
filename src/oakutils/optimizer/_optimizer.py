from __future__ import annotations

import itertools
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Callable

import depthai as dai

from ._grid_search import grid_search

if TYPE_CHECKING:
    from typing_extensions import Self


class Optimizer:
    """Class for optimizing a pipeline onboard an OAK camera."""

    def __init__(
        self: Self,
        algorithm: str = "grid",
        max_measure_time: float = 10.0,
        measure_trials: int = 1,
        warmup_cycles: int = 10,
        stability_threshold: float = 0.002,
        stability_length: int = 10,
    ) -> None:
        """
        Use to create an instance of the class.

        Parameters
        ----------
        algorithm : str, optional
            The algorithm to use for optimization, by default "grid"
            Options are "grid"
        max_measure_time : float, optional
            The amount of time to measure the pipeline for, by default 10.0
        measure_trials : int, optional
            The number of times to measure the pipeline, by default 1
        warmup_cycles : int, optional
            The number of cycles to run before measuring, by default 10
        stability_threshold : float, optional
            The threshold for stability, seconds difference between the max
            and min cycle time during measurement. If the difference is less than
            this threshold, the measurement will stop, by default 0.002 (2 millisecond)
        stability_length : int, optional
            The number of cycles to check for stability, by default 15
            A higher number will typically increase the accuracy of the measurement,
            but will also increase the time it takes to measure.

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
        self._warmup_cycles = warmup_cycles
        self._stability_threshold = stability_threshold
        self._stability_length = stability_length

    def measure(
        self: Self,
        pipeline_func: Callable[
            [dai.Pipeline, dict[str, Any]], list[Callable[[dai.Device], None]]
        ],
        pipeline_args: dict[str, Any],
    ) -> tuple[float, float, dict[str, float]]:
        """
        Use to measure the FPS of a pipeline.

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
        for _ in range(self._measure_trials):
            print(f"Running trial {_ + 1} of {self._measure_trials}")
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
                queues: dict = {
                    q: device.getOutputQueue(name=q, maxSize=1, blocking=False)
                    for q in queue_names
                }

                # run pipelines
                past = deque(maxlen=self._stability_length)
                counter = 0
                t0 = time.perf_counter()
                while not stopped:
                    counter += 1
                    # get start time
                    t1 = time.perf_counter()
                    # handle data
                    for queue_name, queue in queues.items():
                        print(f"      {queue_name}")
                        data = queue.get()
                        if counter >= self._warmup_cycles:
                            data_times[queue_name].append(
                                (dai.Clock.now() - data.getTimestamp()).total_seconds()
                                * 1000
                            )
                    # handle cycles
                    if counter < self._warmup_cycles:
                        continue
                    # get end time
                    t2 = time.perf_counter()
                    cycle_time = t2 - t1
                    cycle_times.append(cycle_time)
                    past.append(cycle_time)
                    elapsed = t2 - t0
                    print(f"   Elapsed: {elapsed:.2f}s")
                    stopped1 = elapsed > self._max_measure_time
                    stopped2 = (
                        max(past) - min(past) < self._stability_threshold
                    ) and len(past) == self._stability_length
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

        print("Computing average stats")
        # compute average fps
        avg_fps = sum(fps) / len(fps)
        avg_latencies = sum(latencies) / len(latencies)
        avg_all_latencies: dict[str, float] = {}
        for key in all_latencies[0]:
            print(f"   {key}")
            data = [latency[key] for latency in all_latencies]
            avg_all_latencies[key] = sum(data) / len(data)
        print("Done")

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
        """
        Use to generate optimized arguments for a pipeline.

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
        # generate all possible assignments
        possible_args: list[dict[str, Any]] = [
            dict(zip(pipeline_args.keys(), values))
            for values in itertools.product(*pipeline_args.values())
        ]
        try:
            # run the selected algorithm for generating measurements
            results = self._algorithm(
                pipeline_func=pipeline_func,
                possible_args=possible_args,
                measure_func=self.measure,
            )
            # run the objective function to get the best arguments
            return objective_func(results)
        except TypeError:
            return self._algorithm(
                pipeline_func=pipeline_func,
                possible_args=possible_args,
                measure_func=self.measure,
                objective_func=objective_func,
            )
