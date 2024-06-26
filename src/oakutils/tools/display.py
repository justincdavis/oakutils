# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Module for creating and using displays for visualization.

Classes
-------
DisplayManager
    Used to display multiple camera streams at once.

Functions
---------
get_resolution_area
    Gets the area of the given resolution.
order_resolutions
    Orders the given resolutions from smallest to largest.
get_smaller_size
    Gets the smaller size of the two given sizes.
"""

from __future__ import annotations

import atexit
from collections import defaultdict
from typing import TYPE_CHECKING, Callable, Iterable

import cv2  # type: ignore[import]
from cv2ext import Display

if TYPE_CHECKING:
    import depthai as dai
    import numpy as np
    from typing_extensions import Self


class DisplayManager:
    """Used in the Camera class to display all the image streams."""

    def __init__(
        self: Self,
        fps: int = 30,
        display_size: tuple[int, int] = (640, 480),
    ) -> None:
        """
        Use to initialize a display manager.

        Parameters
        ----------
        fps : int, optional
            The fps of the display manager, by default 30
        display_size : Tuple[int, int], optional
            The size of the display, by default (640, 480)

        """
        self._displays: dict[str, Display] = {}
        self._transforms: dict[str, Callable] = defaultdict(lambda: lambda x: x)
        self._display_size = display_size
        self._fps = fps
        atexit.register(self._stop)

    @property
    def fps(self: Self) -> int:
        """
        Returns the fps of the display manager.

        Returns
        -------
        int
            The fps of the display manager

        """
        return self._fps

    @fps.setter
    def fps(self: Self, fps: int) -> None:
        """
        Use to set the fps of the display manager.

        Parameters
        ----------
        fps : int
            The fps to set the display manager to

        """
        self._fps = fps

    def _stop(self: Self) -> None:
        for display in self._displays.values():
            display.stop()

    def stop(self: Self) -> None:
        """Use to stop the display manager."""
        self._stop()

    def _update(self: Self, name: str, frame: np.ndarray) -> None:
        if (
            frame.shape[1] != self._display_size[0]
            or frame.shape[0] != self._display_size[1]
        ):
            frame = cv2.resize(
                frame,
                self._display_size,
                interpolation=cv2.INTER_LINEAR,
            )
        try:
            self._displays[name](frame)
        except KeyError:
            self._displays[name] = Display(name, fps=self._fps)
            self._displays[name](frame)

    def set_transform(
        self: Self,
        name: str,
        transform: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        """
        Use to set a transform for the given name.

        A transform should take in an np.ndarray and return an np.ndarray.

        Parameters
        ----------
        name : str
            The name of the transform
        transform : Callable[[np.ndarray], np.ndarray]
            The transform to set

        """
        self._transforms[name] = transform

    def update(
        self: Self,
        data: tuple[str, np.ndarray] | list[tuple[str, np.ndarray]],
        transform: Callable | None = None,
    ) -> None:
        """
        Use to update the display manager with the given data.

        Parameters
        ----------
        data : Union[Tuple[str, np.ndarray], list[str, np.ndarray]]
            The data to update the display with. Can be a single tuple or an
            list of tuples.
        transform : Optional[Callable], optional
            A transform to call on each frame, by default None
            The transform should take in an np.ndarray and return an np.ndarray

        """
        # whether or not we are in Tuple or list case
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

    def callback(self: Self, name: str) -> Callable[[dai.ImgFrame], None]:
        """
        Use to get a callback to be used with ImgFrame outputs.

        The outputs come from queues from a depthai.Device. The callback
        will update the display with the given name based on the data
        from the ImgFrame.

        Parameters
        ----------
        name : str
            The name of the output queue to add the callback to.

        Returns
        -------
        Callable[[dai.ImgFrame], None]
            The callback to be used with the Camera class.

        """

        def callback(frame: dai.ImgFrame) -> None:
            cv_frame: np.ndarray = frame.getCvFrame()  # type: ignore[assignment]
            self._update(name, cv_frame)

        return callback


def get_resolution_area(resolution: tuple[int, int]) -> int:
    """
    Use to get the area of the given resolution.

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
    resolutions: Iterable[tuple[int, int]],
    *,
    reverse: bool | None = None,
) -> list[tuple[int, int]]:
    """
    Use to order the given resolutions from smallest to largest.

    Parameters
    ----------
    resolutions : Iterable[Tuple[int, int]]
        The resolutions to order
    reverse : Optional[bool], optional
        Whether or not to reverse the order, by default False
        If reverse is True, then it will order from largest to smallest.

    Returns
    -------
    list[Tuple[int, int]]
        The ordered resolutions

    """
    if reverse is None:
        reverse = False
    return sorted(resolutions, key=get_resolution_area, reverse=reverse)


def get_smaller_size(size1: tuple[int, int], size2: tuple[int, int]) -> tuple[int, int]:
    """
    Use to get the smaller size of the two given sizes.

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
