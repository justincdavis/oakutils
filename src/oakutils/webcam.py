from __future__ import annotations

import atexit
import contextlib
from threading import Condition, Thread
from typing import TYPE_CHECKING

import depthai as dai

from .nodes import create_color_camera, create_xout
from .tools.parsing import get_color_sensor_resolution_from_tuple

if TYPE_CHECKING:
    import numpy as np
    from typing_extensions import Self


class Webcam:
    def __init__(
        self: Self, resolution: tuple[int, int] = (1920, 1080), fps: int = 30
    ) -> None:
        """Create a new Webcam object.

        Parameters
        ----------
        resolution : tuple[int, int], optional
            The resolution of the webcam, by default (1920, 1080)
        fps : int, optional
            The framerate of the webcam, by default 30
        """
        self._resolution = resolution
        self._fps = fps

        # depthai stuff
        self._pipeline = dai.Pipeline()
        dai_resolution = get_color_sensor_resolution_from_tuple(self._resolution)
        self._cam = create_color_camera(
            self._pipeline, resolution=dai_resolution, fps=self._fps
        )
        self._xout_cam = create_xout(self._pipeline, self._cam.video, "cam")

        # frame storage
        self._frame: np.ndarray = None

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

    def __del__(self: Self) -> None:
        self.stop()

    def stop(self: Self) -> None:
        """Stop the camera."""
        self._stopped = True
        with contextlib.suppress(RuntimeError):
            self._thread.join()

    def read(self: Self) -> tuple[bool, np.ndarray | None]:
        """Read a frame from the camera.

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
            q_camera = device.getOutputQueue(name="cam", maxSize=1, blocking=False)

            # loop until stopped
            while not self._stopped:
                # get data
                self._frame = q_camera.get().getCvFrame()

                # notify that we have started
                if not self._started:
                    with self._start_condition:
                        self._started = True
                        self._start_condition.notify()
