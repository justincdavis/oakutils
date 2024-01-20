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
Module for filtering aruco marker detections as a continous stream.

Classes
-------
ArucoStream
    Use to filter aruco marker detections as a continous stream.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import TYPE_CHECKING

import cv2  # type: ignore[import]

from .finder import ArucoFinder

if TYPE_CHECKING:
    import numpy as np
    from typing_extensions import Self

    from oakutils.calibration import ColorCalibrationData, MonoCalibrationData


class ArucoStream:
    """
    Class for filtering aruco marker detections as a continous stream.

    Attributes
    ----------
    calibration : ColorCalibrationData, MonoCalibrationData, None
        The calibration data to use for finding the transformation matrix

    Methods
    -------
    find(image: np.ndarray, rectified: bool | None = None)
        Finds the aruco markers in the image
    draw(image: np.ndarray, markers: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]])
    Draws the detected markers onto the image
    """

    def __init__(
        self: Self,
        aruco_dict: int = cv2.aruco.DICT_4X4_100,
        marker_size: float = 0.05,
        calibration: ColorCalibrationData | MonoCalibrationData | None = None,
        buffersize: int = 5,
        max_age: int = 5,
        alpha: float = 0.95,
    ) -> None:
        """
        Use to create an ArucoStream object.

        Parameters
        ----------
        aruco_dict : int, optional
            The aruco dictionary to use for finding markers,
              by default cv2.aruco.DICT_4X4_100
        marker_size : float, optional
            The size of the markers in meters, by default 0.05
        calibration : ColorCalibrationData, MonoCalibrationData, optional
            The calibration data to use for finding the transformation matrix,
              by default None
            Will utilize an identity matrix if not provided
        buffersize : int, optional
            The size of the buffer to use for filtering detections,
                by default 5
        max_age : int, optional
            The maximum age of a detection in the buffer,
                by default 5
            If the last detection for an id is older than this, then
                the buffer will be cleared for that id
        alpha : float, optional
            The alpha value to use for exponential smoothing,
                by default 0.8, must be in range [0, 1]
            Smaller values will result in more smoothing

        Raises
        ------
        ValueError
            If alpha is not in range [0, 1]
        """
        self._finder: ArucoFinder = ArucoFinder(aruco_dict, marker_size, calibration)
        self._buffers: dict[
            int,
            deque[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        ] = defaultdict(lambda: deque(maxlen=buffersize))
        self._id_age: dict[int, int] = defaultdict(int)
        self._max_age = max_age
        self._age = 0
        if alpha < 0 or alpha > 1:
            err_msg = "alpha must be in range [0, 1]"
            raise ValueError(err_msg)
        self._alpha1, self._alpha2 = alpha, (1.0 - alpha)

    def find(
        self: Self,
        image: np.ndarray,
        *,
        rectified: bool | None = None,
    ) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Use to find the aruco markers in the image and perform filtering.

        Note:
        Makes an assumption that there is a single marker for each id.

        Parameters
        ----------
        image : np.ndarray
            The image to find the marker in
        rectified : bool, optional
            Whether or not the image is rectified, by default None
            If None will use the calibration data to undistort the image

        Returns
        -------
        list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
            The list of aruco markers found in the image
            Each tuple contains the id, transformation matrix,
              rotation vector, translation vector, and corners
        """
        detections = self._finder.find(image, rectified=rectified)

        # need to clear old detections, if an id hasnt been seen in awhile
        # need to empty its buffer
        if self._age % 3 == 0:
            for tag, age in self._id_age.items():
                if self._age - age > self._max_age:
                    self._buffers[tag].clear()

        # perform exponential smoothing to get the new detections
        new_detections = []
        for tag, og_transform, _, _, corners in detections:
            transform = og_transform.copy()
            # past detections
            past = list(self._buffers[tag])
            for p in past:
                past_transform = p[1]
                transform = self._alpha1 * transform + self._alpha2 * past_transform
            rvec = cv2.Rodrigues(transform[:3, :3])[0]
            tvec = transform[:3, 3]
            new_detections.append((tag, transform, rvec, tvec, corners))

        # update the buffers
        for detection in detections:
            self._buffers[detection[0]].append(detection)

        self._age += 1

        return new_detections

    def draw(
        self: Self,
        image: np.ndarray,
        markers: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        """
        Use to draw the detected markers onto the image.

        Parameters
        ----------
        image : np.ndarray
            The image to draw the markers on
        markers : list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
            The list of aruco markers found in the image
            Each tuple contains the id, transformation matrix,
              rotation vector, translation vector, and corners

        Returns
        -------
        np.ndarray
            A copy of the image with the markers drawn on it.
        """
        return self._finder.draw(image, markers)
