from __future__ import annotations

import atexit
import time
from collections import defaultdict
from threading import Thread
from typing import Callable, Dict, Iterable, Optional, Tuple, Union

import cv2
import numpy as np


class _Display:
    def __init__(self, name: str, fps: int = 15):
        self._name = name
        self._fps = fps
        self._delay_time: float = 1 / fps
        self._frame: Optional[np.ndarray] = None
        self._stopped = False
        self._thread = Thread(target=self._run)
        atexit.register(self.stop)

    def __call__(self, frame: np.ndarray):
        self._frame = frame

    def __del__(self):
        self.stop()

    def stop(self):
        self._stopped = True
        try:
            self._thread.join()
        except RuntimeError:
            pass

    def _run(self):
        while self._stopped:
            if self._frame is not None:
                s = time.time()
                cv2.imshow(self._name, self._frame)
                self._frame = None
                e = time.time()
                cv2.waitKey(max(1, int((self._delay_time - (e - s)) * 1000)))
        cv2.destroyWindow(self._name)


class DisplayManager:
    """
    Used in the Camera class to display all the image streams.
    """

    def __init__(self, fps: int = 15, display_size: Tuple[int, int] = (640, 480)):
        self._displays: Dict[str, _Display] = {}
        self._transforms: Dict[str, Callable] = defaultdict(lambda: lambda x: x)
        self._display_size = display_size
        self._fps = fps
        atexit.register(self._stop)

    def _stop(self):
        for display in self._displays.values():
            display.stop()

    def stop(self):
        """
        Stops the display manager.
        """
        self._stop()

    def _update(self, name: str, frame: np.ndarray):
        if (
            frame.shape[1] != self._display_size[0]
            or frame.shape[0] != self._display_size[1]
        ):
            frame = cv2.resize(frame, self._display_size)
        try:
            self._displays[name](frame)
        except KeyError:
            self._displays[name] = _Display(name, self._fps)
            self._displays[name](frame)

    def set_transform(self, name: str, transform: Callable):
        """
        Sets a transform for the given name.

        Parameters
        ----------
        name : str
            The name of the transform
        transform : Callable
            The transform to set
        """
        self._transforms[name] = transform

    def update(
        self,
        data: Union[Tuple[str, np.ndarray], Iterable[Tuple[str, np.ndarray]]],
        transform: Optional[Callable] = None,
    ):
        """
        Updates the display with the given data.

        Parameters
        ----------
        data : Union[Tuple[str, np.ndarray], Iterable[str, np.ndarray]]
            The data to update the display with. Can be a single tuple or an
            iterable of tuples.
        transform : Optional[Callable], optional
            A transform to call on each frame, by default None
            The transform should take in an np.ndarray and return an np.ndarray
        """
        # whether or not we are in Tuple or Iterable case
        if isinstance(data, tuple):
            name, frame = data
            if transform is not None:
                self.set_transform(name, transform)
            self._update(name, self._transforms[name](frame))
        else:
            name, _ = data[0]
            if transform is not None:
                self.set_transform(name, transform)
            for name, frame in data:
                self._update(name, self._transforms[name](frame))


def get_resolution_area(resolution: Tuple[int, int]) -> int:
    """
    Gets the area of the given resolution.

    Parameters
    ----------
    resolution : Tuple[int, int]
        The resolution to get the area of

    Returns
    -------
    int
        The area of the resolution
    """
    return resolution[0] * resolution[1]


def order_resolutions(
    resolutions: Iterable[Tuple[int, int]], reverse: Optional[bool] = None
) -> Iterable[Tuple[int, int]]:
    """
    Orders the given resolutions from smallest to largest.
    If reverse is True, then it will order from largest to smallest.

    Parameters
    ----------
    resolutions : Iterable[Tuple[int, int]]
        The resolutions to order
    reverse : Optional[bool], optional
        Whether or not to reverse the order, by default False

    Returns
    -------
    Iterable[Tuple[int, int]]
        The ordered resolutions
    """
    if reverse is None:
        reverse = False
    return sorted(resolutions, key=get_resolution_area, reverse=reverse)


def get_smaller_size(size1: Tuple[int, int], size2: Tuple[int, int]) -> Tuple[int, int]:
    """
    Gets the smaller size of the two given sizes.

    Parameters
    ----------
    size1 : Tuple[int, int]
        The first size
    size2 : Tuple[int, int]
        The second size

    Returns
    -------
    Tuple[int, int]
        The smaller size
    """
    return order_resolutions([size1, size2])[0]
