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
Module defining a webcam class for reading frames from an OAK.

Classes
-------
Webcam
    A class for reading frames from an OAK using the same interface as cv2.VideoCapture.
"""
from __future__ import annotations

import atexit
import contextlib
from threading import Condition, Thread
from typing import TYPE_CHECKING

import depthai as dai
import numpy as np

from .calibration import ColorCalibrationData, get_camera_calibration_basic
from .nodes import create_color_camera, create_xout
from .tools.parsing import get_color_sensor_resolution_from_tuple

if TYPE_CHECKING:
    from typing_extensions import Self


class Webcam:
    """
    A class for reading frames from an OAK using the same interface as cv2.VideoCapture.

    Attributes
    ----------
    calibration : ColorCalibrationData
        The calibration info for the camera


    Methods
    -------
    stop()
        Stop the camera.
    read()
        Read a frame from the camera.
    """

    def __init__(
        self: Self,
        resolution: tuple[int, int] = (1920, 1080),
        fps: int = 30,
    ) -> None:
        """
        Create a new Webcam object.

        Parameters
        ----------
        resolution : tuple[int, int], optional
            The resolution of the webcam, by default (1920, 1080)
        fps : int, optional
            The framerate of the webcam, by default 30
        """
        self._resolution = resolution
        self._fps = fps

        # get the calibration
        self._calibration = get_camera_calibration_basic(
            rgb_size=self._resolution,
        )

        # depthai stuff
        self._pipeline = dai.Pipeline()
        dai_resolution = get_color_sensor_resolution_from_tuple(self._resolution)
        self._cam = create_color_camera(
            self._pipeline,
            resolution=dai_resolution,
            fps=self._fps,
        )
        self._xout_cam = create_xout(self._pipeline, self._cam.video, "cam")

        # frame storage
        self._frame: np.ndarray = np.zeros((640, 480, 3), dtype=np.uint8)

        # thread for reading camera
        self._started = False
        self._stopped: bool = False
        self._thread: Thread = Thread(target=self._run, daemon=True)
        self._start_condition = Condition()

        # register stop function
        atexit.register(self.stop)

        # start the camera
        self._thread.start()
        with self._start_condition:
            self._start_condition.wait()

    @property
    def calibration(self: Self) -> ColorCalibrationData:
        """
        Returns the calibration info for the camera.

        Returns
        -------
        ColorCalibrationData
            The calibration info for the camera
        """
        return self._calibration.rgb

    def __del__(self: Self) -> None:
        """Stop the camera when the object is deleted."""
        self.stop()

    def stop(self: Self) -> None:
        """Stop the camera."""
        self._stopped = True
        with contextlib.suppress(RuntimeError):
            self._thread.join()

    def read(self: Self) -> tuple[bool, np.ndarray | None]:
        """
        Read a frame from the camera.

        Returns
        -------
        tuple[bool, np.ndarray]
            A tuple containing a boolean indicating if the frame was read successfully and the frame itself.
        """
        # get data
        return self._frame is not None, self._frame

    def _run(self: Self) -> None:
        """Run the camera."""
        with dai.Device(self._pipeline) as device:
            # get data queues
            q_camera = device.getOutputQueue(name="cam", maxSize=1, blocking=False)  # type: ignore[attr-defined]

            # loop until stopped
            while not self._stopped:
                # get data
                self._frame = q_camera.get().getCvFrame()

                # notify that we have started
                if not self._started:
                    with self._start_condition:
                        self._started = True
                        self._start_condition.notify()
