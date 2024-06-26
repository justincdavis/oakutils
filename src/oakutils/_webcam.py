# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
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

from .calibration import ColorCalibrationData, get_oak1_calibration
from .core import create_device
from .nodes import create_color_camera, create_xout
from .tools.parsing import get_color_sensor_resolution_from_tuple

if TYPE_CHECKING:
    from typing_extensions import Self


class Webcam:
    """A class for reading frames from an OAK using the same interface as cv2.VideoCapture."""

    def __init__(
        self: Self,
        resolution: tuple[int, int] = (1920, 1080),
        fps: int = 30,
        device_id: str | None = None,
    ) -> None:
        """
        Create a new Webcam object.

        Parameters
        ----------
        resolution : tuple[int, int], optional
            The resolution of the webcam, by default (1920, 1080)
        fps : int, optional
            The framerate of the webcam, by default 30
        device_id : str, optional
            The id of the device to use, by default None
            This can be a MXID, IP address, or USB port name.
            Examples: "14442C108144F1D000", "192.168.1.44", "3.3.3"

        """
        self._resolution = resolution
        self._fps = fps
        self._mxid: str | None = device_id

        # get the calibration
        self._calibration = get_oak1_calibration(
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
        return self._calibration

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
        device_object = create_device(self._pipeline, device_id=self._mxid)
        with device_object as device:
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
