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
Module for creating custom pipelines using a lightweight callback based class.

Classes
-------
ApiCamera
    A lightweight class for creating custom pipelines using callbacks.
"""
from __future__ import annotations

import atexit
import contextlib
from functools import partial
from threading import Condition, Thread
from typing import TYPE_CHECKING, Callable, Iterable

import depthai as dai

from .calibration import CalibrationData, get_camera_calibration
from .point_clouds import PointCloudVisualizer
from .tools.display import DisplayManager, get_smaller_size

if TYPE_CHECKING:
    from typing_extensions import Self


class ApiCamera:
    """
    A lightweight class for creating custom pipelines using callbacks.

    Attributes
    ----------
    pipeline:
        The pipeline for the camera
    calibration : CalibrationData
        The calibration info for the camera
    displays : DisplayManager
        The display manager for the camera
    pcv : PointCloudVisualizer
        The point cloud visualizer for the camera

    Methods
    -------
    start(blocking=False)
        Start the camera.
    stop()
        Stop the camera.
    add_callback(name, callback)
        Add a callback to the camera.
    add_display(name)
        Add a display callback to the camera.
    add_device_call(call)
        Add a device call to the camera.
    """

    def __init__(
        self: Self,
        # custom args, only related to configuration
        color_size: tuple[int, int] = (1920, 1080),
        mono_size: tuple[int, int] = (640, 400),
        *,
        primary_mono_left: bool | None = None,
    ) -> None:
        """
        Use to create an instance of the camera.

        Parameters
        ----------
        primary_mono_left : bool, optional
            Whether the primary mono camera is on the left or not, by default None
        color_size : tuple[int, int], optional
            The size of the color camera, by default (1920, 1080)
        mono_size : tuple[int, int], optional
            The size of the mono camera, by default (640, 400)
        """
        if primary_mono_left is None:
            primary_mono_left = True

        # store custom args
        self._color_size: tuple[int, int] = color_size
        self._mono_size: tuple[int, int] = mono_size
        self._primary_mono_left: bool = primary_mono_left

        # handle attributes
        self._calibration: CalibrationData = get_camera_calibration(
            rgb_size=self._color_size,
            mono_size=self._mono_size,
            is_primary_mono_left=self._primary_mono_left,
        )
        self._callbacks: dict[str | Iterable[str], Callable] = {}
        self._pipeline: dai.Pipeline = dai.Pipeline()
        self._is_built: bool = False
        self._custom_device_calls: list[Callable[[dai.DeviceBase], None]] = []

        # handle custom displays directly for API stuff without visualize
        self._display_size: tuple[int, int] = get_smaller_size(
            self._color_size,
            self._mono_size,
        )
        self._displays: DisplayManager | None = None
        self._pcv: PointCloudVisualizer | None = None

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

    def __del__(self: Self) -> None:
        """Use to stop the camera."""
        self.stop()

    @property
    def pipeline(self: Self) -> dai.Pipeline:
        """
        Use to get the pipeline. This is useful for adding custom nodes to the pipeline.

        Raises
        ------
        RuntimeError
            If the pipeline is accessed once the camera has been started.
        """
        if self._started:
            err_msg = "Cannot access pipeline once camera has been started."
            raise RuntimeError(err_msg)
        return self._pipeline

    @property
    def calibration(self: Self) -> CalibrationData:
        """Use to get the calibration data."""
        return self._calibration

    @property
    def displays(self: Self) -> DisplayManager:
        """Use to get the display manager."""
        if self._displays is None:
            self._displays = DisplayManager(display_size=self._display_size)
        return self._displays

    @property
    def pcv(self: Self) -> PointCloudVisualizer:
        """Use to get the point cloud visualizer."""
        if self._pcv is None:
            self._pcv = PointCloudVisualizer(window_size=self._display_size)
        return self._pcv

    def start(self: Self, *, blocking: bool | None = None) -> None:
        """Use to start the camera. To be done after all api calls are made."""
        if blocking is None:
            blocking = False

        with self._start_condition:
            self._start_condition.notify()
        self._started = True

        if blocking:
            with self._stop_condition:
                self._stop_condition.wait()

    def stop(self: Self) -> None:
        """Use to stop the camera."""
        self._stopped = True

        # call conditions if system never started
        with self._start_condition:
            self._start_condition.notify_all()

        with contextlib.suppress(RuntimeError):
            self._thread.join()

    def add_callback(self: Self, name: str | Iterable[str], callback: Callable) -> None:
        """
        Use to add a callback to be run on the output queue with the given name.

        Parameters
        ----------
        name : str
            The name of the output queue to add the callback to.
        callback : Callable
            The callback to add.
        """
        self._callbacks[name] = callback

    def add_display(self: Self, name: str) -> None:
        """
        Use to add a display callback for the given stream name.

        Parameters
        ----------
        name : str
            The name of the output queue to add the callback to.
        """
        self.add_callback(name, self.displays.callback(name))

    def add_device_call(self: Self, call: Callable[[dai.DeviceBase], None]) -> None:
        """
        Use to add a device call to be run after the device is created.

        Parameters
        ----------
        call : Callable[[dai.Device], None]
            The call to add.

        Raises
        ------
        RuntimeError
            If the camera has already been started.
        """
        if self._started:
            err_msg = "Cannot add device call after camera has been started."
            raise RuntimeError(err_msg)
        self._custom_device_calls.append(call)

    def _run(self: Self) -> None:
        # wait for the start call, this allows user to define pipeline
        with self._start_condition:
            self._start_condition.wait()

        with dai.Device(self._pipeline) as device:
            # run any custom devices calls added ahead of time
            for custom in self._custom_device_calls:
                custom(device)

            # get the output queues ahead of time
            queues = {
                name: device.getOutputQueue(name) for name, _ in self._callbacks.items()  # type: ignore[attr-defined]
            }

            # create a cache for queue results to enable multi queue callbacks
            data_cache = {name: None for name, _ in self._callbacks.items()}

            while not self._stopped:
                # cache results
                for name in data_cache:
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
                for callback in partials:
                    callback()

        # call stop conditions if start was called with blocking
        with self._stop_condition:
            self._stop_condition.notify()
