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
Module for localizing the camera in the world using ArUco markers.

Classes
-------
ArucoLocalizer
    Use to localize the camera in the world using ArUco markers.
"""
from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from oakutils.tools.transform import create_transform

if TYPE_CHECKING:
    from typing_extensions import Self


class ArucoLocalizer:
    """
    Localizes the camera in the world using ArUco markers.

    Methods
    -------
    add_transform(tag: int, transform: np.ndarray)
        Use to add a transform to the localizer.
    localize(markers: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]])
        Use to localize the camera in the world using ArUco markers.
    """

    def __init__(
        self: Self,
        transforms: dict[int, np.ndarray],
        buffersize: int = 5,
        max_age: int = 5,
        alpha: float = 0.95,
    ) -> None:
        """
        Use to create a new ArucoLocalizer.

        Parameters
        ----------
        transforms : dict[int, np.ndarray]
            A dictionary of transforms from the camera to the marker
        buffersize : int, optional
            The size of the buffer to use for filtering transforms,
                by default 5
        max_age : int, optional
            The maximum age of a detection in the buffer,
                by default 5
        alpha : float, optional
            The alpha value to use for exponential smoothing,
        by default 0.95, must be in range [0, 1]
        """
        self._transforms: dict[int, np.ndarray] = {}
        min_alpha, max_alpha = 0.0, 1.0
        if alpha < min_alpha or alpha > max_alpha:
            err_msg = "alpha must be in range [0, 1]"
            raise ValueError(err_msg)
        self._alpha1, self._alpha2 = alpha, (1.0 - alpha)
        self._max_age = max_age
        self._age = 0
        self._buffer: deque[np.ndarray] = deque(maxlen=buffersize)
        for tag, transform in transforms.items():
            self.add_transform(tag, transform)
        self._last_transform = create_transform(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def add_transform(self: Self, tag: int, transform: np.ndarray) -> None:
        """
        Use to add a transform to the localizer.

        Parameters
        ----------
        tag : int
            The id of the marker to use
        transform : np.ndarray
            The transform from the world to the marker
        """
        self._transforms[tag] = transform

    def localize(
        self: Self,
        markers: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        """
        Use to localize the camera in the world using ArUco markers.

        Parameters
        ----------
        markers : list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
            A list of markers detected sin the image
        """
        transforms = []
        for tag, transform, _, _, _ in markers:
            if tag in self._transforms:
                # marker transform is world to marker
                marker_transform = self._transforms[tag]
                # our transform is camera to marker
                transforms.append(marker_transform.dot(np.linalg.inv(transform)))
        if len(transforms) == 0:
            self._age += 1  # increment ages
            if self._age > self._max_age:
                self._buffer.clear()
            return self._last_transform

        og_transform: np.ndarray = np.mean(transforms, axis=0)
        transform = og_transform.copy()
        for past_transform in list(self._buffer):
            transform = self._alpha1 * transform + self._alpha2 * past_transform
        self._last_transform = transform
        self._buffer.append(og_transform)

        self._age = 0  # reset age since we found markers

        return transform
