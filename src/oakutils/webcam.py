from __future__ import annotations

import atexit
from threading import Thread, Condition

import depthai as dai
import numpy as np

from .nodes import create_color_camera, create_xout
from .tools.parsing import get_color_resolution_from_tuple


class Webcam:
    def __init__(self, resolution: tuple[int, int] = (1920, 1080), fps: int = 30) -> None:
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
        dai_resolution = get_color_resolution_from_tuple(self._resolution)
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
        self._start_condition: Condition = Condition()

        # register stop function
        atexit.register(self.stop)

        # start the camera
        self._thread.start()

    def __del__(self):
        self.stop()

    def stop(self) -> None:
        """Stop the camera."""
        self._stopped = True
        self._thread.join()

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read a frame from the camera.

        Returns
        -------
        tuple[bool, np.ndarray]
            A tuple containing a boolean indicating if the frame was read successfully and the frame itself.
        """
        if not self._thread.is_alive():
            return False, None

        # wait for the camera to be ready
        if not self._started:
            with self._start_condition:
                self._start_condition.wait()

        # get data
        return True, self._frame
    
    def _run(self) -> None:
        """Run the camera."""
        with dai.Device(self._pipeline) as device:
            # get data queues
            q_camera = device.getOutputQueue(name="cam", maxSize=1, blocking=False)

            # notify that the camera is ready
            with self._start_condition:
                self._started = True
                self._start_condition.notify_all()

            # loop until stopped
            while not self._stopped:
                # get data
                self._frame = q_camera.get().getCvFrame()
    