from __future__ import annotations

import atexit
import contextlib
import time
from collections import defaultdict
from threading import Thread
from typing import TYPE_CHECKING, Callable, Iterable

import cv2

if TYPE_CHECKING:
    import numpy as np
    from typing_extensions import Self


class _Display:
    def __init__(self: Self, name: str, fps: int = 15) -> None:
        self._name = name
        self._fps = fps
        self._delay_time: float = 1 / fps
        self._frame: np.ndarray | None = None
        self._stopped = False
        self._thread = Thread(target=self._run)
        atexit.register(self.stop)

    def __call__(self: Self, frame: np.ndarray) -> None:
        self._frame = frame

    def __del__(self: Self) -> None:
        self.stop()

    def stop(self: Self) -> None:
        self._stopped = True
        with contextlib.suppress(RuntimeError):
            self._thread.join()

    def _run(self: Self) -> None:
        while self._stopped:
            if self._frame is not None:
                s = time.time()
                cv2.imshow(self._name, self._frame)
                self._frame = None
                e = time.time()
                cv2.waitKey(max(1, int((self._delay_time - (e - s)) * 1000)))
        cv2.destroyWindow(self._name)


class DisplayManager:
    """Used in the Camera class to display all the image streams."""

    def __init__(
        self: Self, fps: int = 15, display_size: tuple[int, int] = (640, 480)
    ) -> None:
        self._displays: dict[str, _Display] = {}
        self._transforms: dict[str, Callable] = defaultdict(lambda: lambda x: x)
        self._display_size = display_size
        self._fps = fps
        atexit.register(self._stop)

    def _stop(self: Self) -> None:
        for display in self._displays.values():
            display.stop()

    def stop(self: Self) -> None:
        """Stops the display manager."""
        self._stop()

    def _update(self: Self, name: str, frame: np.ndarray) -> None:
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

    def set_transform(self: Self, name: str, transform: Callable) -> None:
        """Sets a transform for the given name.

        Parameters
        ----------
        name : str
            The name of the transform
        transform : Callable
            The transform to set
        """
        self._transforms[name] = transform

    def update(
        self: Self,
        data: tuple[str, np.ndarray] | Iterable[tuple[str, np.ndarray]],
        transform: Callable | None = None,
    ) -> None:
        """Updates the display with the given data.

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


def get_resolution_area(resolution: tuple[int, int]) -> int:
    """Gets the area of the given resolution.

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
    resolutions: Iterable[tuple[int, int]], reverse: bool | None = None
) -> Iterable[tuple[int, int]]:
    """Orders the given resolutions from smallest to largest.
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


def get_smaller_size(size1: tuple[int, int], size2: tuple[int, int]) -> tuple[int, int]:
    """Gets the smaller size of the two given sizes.

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
