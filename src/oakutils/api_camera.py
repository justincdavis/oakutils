from __future__ import annotations

from threading import Thread, Condition
from typing import Tuple, Optional, Union, Dict, Callable, Iterable
import atexit
from functools import partial

import depthai as dai

from .calibration import CalibrationData, get_camera_calibration
from .point_clouds import PointCloudVisualizer
from .tools.display import DisplayManager, get_smaller_size


class Camera:
    def __init__(
        self,
        # custom args, only related to configuration
        primary_mono_left: bool = True,
        color_size: Tuple[int, int] = (1920, 1080),
        mono_size: Tuple[int, int] = (640, 400),
    ):
        # store custom args
        self._color_size: Tuple[int, int] = color_size
        self._mono_size: Tuple[int, int] = mono_size
        self._primary_mono_left: bool = primary_mono_left

        # handle attributes
        self._calibration: CalibrationData = get_camera_calibration(
            rgb_size=self._color_size, mono_size=self._mono_size, primary_mono_left=self._primary_mono_left
        )
        self._callbacks: Dict[Union[str, Iterable[str]], Callable] = {}
        self._pipeline: Optional[dai.Pipeline] = dai.Pipeline()
        self._is_built: bool = False

        # handle custom displays directly for API stuff without visualize
        self._display_size: Tuple[int, int] = get_smaller_size(
            self._color_size, self._mono_size
        )
        self._displays: Optional[DisplayManager] = None
        self._pcv: Optional[PointCloudVisualizer] = None

        # thread for reading camera
        self._started = False
        self._stopped: bool = False
        self._thread: Thread = Thread(target=self._run, daemon=True)
        self._start_condition: Condition = Condition()
        self._stop_condition: Condition = Condition()
        self._intialize_condition: Condition = Condition()
        self._thread.start()

        # register stop function
        atexit.register(self.stop)

        # wait for the camera to be ready
        with self._intialize_condition:
            self._intialize_condition.wait()

    def __del__(self):
        self.stop()

    @property
    def pipeline(self) -> dai.Pipeline:
        """
        Returns the pipeline. If the pipeline has not been built yet, a RuntimeError is raised.
        This is useful for adding custom nodes to the pipeline.

        Raises
        ------
        RuntimeError
            If the pipeline is accessed once the camera has been started.
        """
        if self._started:
            raise RuntimeError("Cannot access pipeline once camera has been started.")
        return self._pipeline

    @property
    def calibration(self) -> CalibrationData:
        """
        Returns the calibration data.
        """
        return self._calibration

    @property
    def displays(self) -> DisplayManager:
        """
        Returns the display manager.
        """
        if self._displays is None:
            self._displays = DisplayManager(display_size=self._display_size)
        return self._displays

    @property
    def pcv(self) -> PointCloudVisualizer:
        """
        Returns the point cloud visualizer.
        """
        if self._pcv is None:
            self._pcv = PointCloudVisualizer(window_size=self._display_size)
        return self._pcv

    def start(self, blocking: bool = False):
        """
        Starts the camera. To be done after all api calls are made.
        Will build the pipeline if it has not been built yet.
        """
        with self._start_condition:
            self._start_condition.notify()
        self._started = True

        if blocking:
            with self._stop_condition:
                self._stop_condition.wait()

    def stop(self):
        """
        Stops the camera.
        """
        self._stopped = True

        # call conditions if system never started
        with self._start_condition:
            self._start_condition.notify_all()

        try:
            self._thread.join()
        except RuntimeError:
            pass

    def add_callback(self, name: Union[str, Iterable[str]], callback: Callable):
        """
        Adds a callback to be run on the output queue with the given name.

        Parameters
        ----------
        name : str
            The name of the output queue to add the callback to.
        callback : Callable
            The callback to add.
        """
        self._callbacks[name] = callback

    def _run(self):
        with self._intialize_condition:
            self._intialize_condition.notify()

        # wait for the start call, this allows user to define pipeline
        with self._start_condition:
            self._start_condition.wait()

        with dai.Device(self._pipeline) as device:
            # get the output queues ahead of time
            queues = {
                name: device.geatOutputQueue(name)
                for name, _ in self._callbacks.items()
            }
            # create a cache for queue results to enable multi queue callbacks
            data_cache = {name: None for name, _ in self._callbacks.items()}
            while not self._stopped:
                # cache results
                for name in data_cache.keys():
                    data_cache[name] = queues[name].get()
                # create callback partials
                partials = []
                for name, callback in self._callbacks.items():
                    if isinstance(name, str):
                        data = data_cache[name]
                    else:
                        data = [data_cache[n] for n in name]
                    partials.append(partial(callback, data))
                # run/dispatch the callback partials
                # TODO: run in async loop or another thread or process?
                for callback in partials:
                    callback()

        # call stop conditions if start was called with blocking
        with self._stop_condition:
            self._stop_condition.notify()
