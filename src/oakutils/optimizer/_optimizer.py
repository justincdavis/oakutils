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
from __future__ import annotations

import itertools
import logging
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Callable

import depthai as dai

from ._grid_search import grid_search

if TYPE_CHECKING:
    from typing_extensions import Self

_log = logging.getLogger(__name__)


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
        if algorithm != "grid":
            err_msg = f"Invalid algorithm {algorithm}"
            raise ValueError(err_msg)
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
            [dai.Pipeline, dict[str, Any]],
            list[Callable[[dai.DeviceBase], None]],
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
        for trial in range(self._measure_trials):
            _log.debug(f"Running trial {trial + 1} of {self._measure_trials}")
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
                queue_names: list[str] = device.getOutputQueueNames()  # type: ignore[attr-defined]
                for q_name in queue_names:
                    data_times[q_name] = []
                queues: dict[str, dai.DataOutputQueue] = {
                    q: device.getOutputQueue(name=q)  # type: ignore[attr-defined]
                    for q in queue_names
                }

                # run pipelines
                past: deque[float] = deque(maxlen=self._stability_length)
                counter = 0
                t0 = time.perf_counter()
                device_t0 = dai.Clock.now().total_seconds()  # type: ignore[call-arg]
                while not stopped:
                    counter += 1
                    # get start time
                    t1 = time.perf_counter()
                    # handle data
                    for queue_name, queue in queues.items():
                        _log.debug(f"      {queue_name}")
                        data = queue.get()
                        data.getData()  # type: ignore[attr-defined]
                        if counter >= self._warmup_cycles:
                            current: float = dai.Clock.now().total_seconds()  # type: ignore[call-arg]
                            data_ts: float = data.getTimestampDevice().total_seconds()  # type: ignore[attr-defined]
                            _log.debug(
                                f"Base: {device_t0}, Curr: {current}, Data: {data_ts}",
                            )
                            ts: float = current - device_t0 - data_ts
                            _log.debug(f"Measured latency: {ts:.2f}ms for {queue_name}")
                            data_times[queue_name].append(ts)
                    # handle cycles
                    if counter < self._warmup_cycles:
                        continue
                    # get end time
                    t2 = time.perf_counter()
                    cycle_time = t2 - t1
                    _log.debug(f"Measured cycle time: {cycle_time:.2f}ms")
                    cycle_times.append(cycle_time)
                    past.append(cycle_time)
                    elapsed = t2 - t0
                    _log.debug(f"Trial: {trial + 1}, Elapsed: {elapsed:.2f}s")
                    stopped1 = elapsed > self._max_measure_time
                    stopped2 = (
                        max(past) - min(past) < self._stability_threshold
                    ) and len(past) == self._stability_length
                    stopped = stopped1 or stopped2

            # compute average latency
            avg_latencies: dict[str, float] = {}
            for queue_name, data_list in data_times.items():
                avg_latencies[queue_name] = sum(data_list) / len(data_list)
            avg_latency = sum(avg_latencies.values()) / len(avg_latencies)

            # compute average cycle time
            avg_cycle_time = sum(cycle_times) / len(cycle_times)

            fps.append(1000.0 / avg_cycle_time)
            latencies.append(avg_latency)
            all_latencies.append(avg_latencies)

        _log.debug(
            f"Done measuring, computing results on {self._measure_trials} trials",
        )
        # compute average fps
        avg_fps = sum(fps) / len(fps)
        avg_latencies_final = sum(latencies) / len(latencies)
        avg_all_latencies: dict[str, float] = {}
        for key in all_latencies[0]:
            _log.debug(f"   {key}")
            key_data = [latency[key] for latency in all_latencies]
            avg_all_latencies[key] = sum(key_data) / len(key_data)

        return avg_fps, avg_latencies_final, avg_all_latencies

    def optimize(
        self: Self,
        pipeline_func: Callable[
            [dai.Pipeline, dict[str, Any]],
            list[Callable[[dai.DeviceBase], None]],
        ],
        pipeline_args: dict[str, list[Any]],
        objective_func: Callable[
            [list[tuple[tuple[float, float, dict[str, float]], dict[str, Any]]]],
            tuple[dict[str, Any], tuple[float, float, dict[str, float]]],
        ],
    ) -> tuple[dict[str, Any], tuple[float, float, dict[str, float]]]:
        """
        Use to generate optimized arguments for a pipeline.

        Parameters
        ----------
        pipeline_func : Callable[[dai.Pipeline, dict[str, Any]], list[Callable[[dai.DeviceBase], None]]]
            The function to generate a pipeline
        pipeline_args : dict[str, list[Any]]
            The arguments to optimize
        objective_func : Callable[[list[tuple[tuple[float, float, dict[str, float]], dict[str, Any]]]], dict[str, Any]]
            The function to use to choose the best arguments

        Returns
        -------
        dict[str, Any]
            The optimized arguments
        tuple[float, float, dict[str, float]]
            The best measurement results
        """
        # generate all possible assignments
        possible_args: list[dict[str, Any]] = [
            dict(zip(pipeline_args.keys(), values))
            for values in itertools.product(*pipeline_args.values())
        ]
        return self._algorithm(
            pipeline_func=pipeline_func,
            possible_args=possible_args,
            measure_func=self.measure,
            objective_func=objective_func,
        )
