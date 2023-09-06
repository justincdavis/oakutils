from threading import Thread, Condition
from typing import List, Tuple, Optional, Union, Dict, Callable, Iterable
import atexit
from functools import partial

import depthai as dai
import depthai_sdk as sdk
import numpy as np
import cv2
import open3d as o3d

from .calibration import CalibrationData, get_camera_calibration
from .point_clouds import (
    PointCloudVisualizer,
    get_point_cloud_from_rgb_depth_image,
    filter_point_cloud,
)
from .nodes import (
    create_color_camera,
    create_stereo_depth,
    create_imu,
)
from .tools.parsing import (
    get_color_sensor_info_from_str,
    get_mono_sensor_info_from_str,
    get_median_filter_from_str,
)
from .tools.display import DisplayManager, get_smaller_size


class OakCamera:
    def __init__(
        self,
        # standard sdk.OakCamera args
        device: Optional[str] = None,
        usb_speed: Optional[Union[str, dai.UsbSpeed]] = None,
        replay: Optional[str] = None,
        rotation: Optional[int] = None,
        config: Optional[dai.Device.Config] = None,
        # custom args
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
            self._color_size, self._mono_size, self._primary_mono_left
        )
        self._callbacks: Dict[Union[str, Iterable[str]], Callable] = {}
        self._pipeline: Optional[dai.Pipeline] = None
        self._is_built: bool = False

        # store oak camera
        self._oak_camera_args = {
            "device": device,
            "usb_speed": usb_speed,
            "replay": replay,
            "rotation": rotation,
            "config": config,
            "args": True,
        }
        self._oak: Optional[sdk.OakCamera] = None
        self._pipeline: Optional[dai.Pipeline] = None

        # handle custom displays directly for API stuff without visualize
        self._display_size: Tuple[int, int] = get_smaller_size(
            self._color_size, self._mono_size
        )
        self._displays: Optional[DisplayManager] = None
        self._pcv: Optional[PointCloudVisualizer] = None

        # thread for reading camera
        self._built = False
        self._started = False
        self._stopped: bool = False
        self._thread: Thread = Thread(target=self._run, daemon=True)
        self._build_condition: Condition = Condition()
        self._start_condition: Condition = Condition()
        self._stop_condition: Condition = Condition()
        self._thread.start()

        # register stop function
        atexit.register(self.stop)

    def __del__(self):
        self.stop()

    @property
    def oak(self) -> sdk.OakCamera:
        """
        Returns the underlying OakCamera object.

        Raises
        ------
        RuntimeError
            If the OakCamera has not been built yet.
        """
        if self._oak is None:
            raise RuntimeError(
                "OakCamera has not been built yet. Failure in processing thread."
            )
        return self._oak

    @property
    def pipeline(self) -> dai.Pipeline:
        """
        Returns the pipeline. If the pipeline has not been built yet, a RuntimeError is raised.
        This is useful for adding custom nodes to the pipeline.

        Raises
        ------
        RuntimeError
            If the pipeline has not been built yet.
        """
        if self._pipeline is None:
            raise RuntimeError(
                "Pipeline has not been built yet. Failure in depthai_sdk.OakCamera.start() or in processing thread."
            )
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

    def build(self):
        """
        Builds the pipeline. To be done after all sdk calls are made.
        """
        with self._build_condition:
            self._build_condition.notify()

    def start(self, blocking: bool = False):
        """
        Starts the camera. To be done after all api calls are made.
        """
        with self._start_condition:
            self._start_condition.notify()

        if blocking:
            with self._stop_condition:
                self._stop_condition.wait()

    def stop(self):
        """
        Stops the camera.
        """
        self._stopped = True

        # call conditions if system never started
        with self._build_condition:
            self._build_condition.notify()
        with self._start_condition:
            self._start_condition.notify()

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
        with sdk.OakCamera(*self._oak_camera_args) as oak:
            self._cam = oak

            # wait for the build call, this allows user to define sdk calls
            with self._build_condition:
                self._build_condition.wait()
            self._built = True

            # build sdk pipeline
            self._pipeline = oak.build()

            # wait for the start call, this allows user to define pipeline
            with self._start_condition:
                self._start_condition.wait()
            self._started = True

            # start the camera and run the pipeline
            oak.start()

            # get the output queues ahead of time
            queues = {
                name: oak.device.getOutputQueue(name)
                for name, _ in self._callbacks.items()
            }
            # create a cache for queue results to enable multi queue callbacks
            data_cache = {name: None for name, _ in self._callbacks.items()}
            while not self._stopped:
                # poll the camera to get new data
                oak.poll()
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
