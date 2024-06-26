# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Module for finding aruco markers in images and acquiring transformation matrices to them.

Classes
-------
ArucoFinder
    Use to find ArUco markers in an image.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2  # type: ignore[import]
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self

    from oakutils.calibration import ColorCalibrationData, MonoCalibrationData


class ArucoFinder:
    """Class for finding aruco markers in images and acquiring transformation matrices to them."""

    def __init__(
        self: Self,
        aruco_dict: int = cv2.aruco.DICT_4X4_100,
        marker_size: float = 0.05,
        calibration: ColorCalibrationData | MonoCalibrationData | None = None,
    ) -> None:
        """
        Use to create the ArucoFinder class.

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

        """
        self._adict = cv2.aruco.getPredefinedDictionary(aruco_dict)
        self._marker_size = marker_size
        self._calibration = calibration
        self._K: np.ndarray = np.zeros((3, 3), dtype=np.float32)
        self._D: np.ndarray = np.zeros((5, 1), dtype=np.float32)
        if self._calibration is not None:
            self._K = self._calibration.K
            self._D = self._calibration.D

    @property
    def calibration(self: Self) -> ColorCalibrationData | MonoCalibrationData | None:
        """
        The calibration data used by the ArucoFinder.

        Returns
        -------
        ColorCalibrationData, MonoCalibrationData, None
            The calibration data used by the ArucoFinder

        """
        return self._calibration

    @calibration.setter
    def calibration(
        self: Self,
        calibration: ColorCalibrationData | MonoCalibrationData,
    ) -> None:
        """
        Use to set the calibration data used by the ArucoFinder.

        Parameters
        ----------
        calibration : ColorCalibrationData, MonoCalibrationData
            The calibration data to use for finding the transformation matrix

        """
        self._calibration = calibration
        self._K = self._calibration.K
        self._D = self._calibration.D

    def find(
        self: Self,
        image: np.ndarray,
        *,
        rectified: bool | None = None,
    ) -> Sequence[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Use to find the aruco markers in the image.

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
        Sequence[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
            The sequence of aruco markers found in the image
            Each tuple contains the id, transformation matrix,
            rotation vector, translation vector, and corners

        """
        if rectified is None:
            rectified = False
        if not rectified:
            image = cv2.undistort(
                image,
                self._K,
                self._D,
                None,
                self._K,
            )
        marker_data = cv2.aruco.detectMarkers(image, self._adict)  # type: ignore[attr-defined]
        corners: Sequence[np.ndarray] = marker_data[0]
        ids: list[np.ndarray] = marker_data[1]
        ret_val: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for idx, corner in enumerate(corners):
            pose_data = cv2.aruco.estimatePoseSingleMarkers(  # type: ignore[attr-defined]
                [corner],  # type: ignore[list-item]
                self._marker_size,
                self._K,
                self._D,
            )
            rvecs: list[np.ndarray] = pose_data[0]
            tvecs: list[np.ndarray] = pose_data[1]
            try:
                rvec = rvecs[0]
                tvec = tvecs[0]
            except IndexError:
                continue
            r_matrix, _ = cv2.Rodrigues(rvec)  # Get equivalent 3x3 rotation matrix
            t_vector = tvec.T  # Get translation as a 3x1 vector
            t_matrix = np.block([[r_matrix, t_vector], [np.zeros((1, 3)), 1]])

            ret_val.append(
                (ids[idx][0], t_matrix, rvec, tvec, np.array(corner, dtype=np.int32)),
            )

        return ret_val

    def draw(
        self: Self,
        image: np.ndarray,
        markers: Sequence[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        """
        Use to draw the detected markers onto the image.

        Parameters
        ----------
        image : np.ndarray
            The image to draw the markers on
        markers : Sequence[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
            The sequence of aruco markers found in the image
            Each tuple contains the id, transformation matrix,
            rotation vector, translation vector, and corners

        Returns
        -------
        np.ndarray
            A copy of the image with the markers drawn on it

        """
        image = image.copy()
        for marker in markers:
            marker_id, _, rvec, tvec, corner = marker
            cv2.drawFrameAxes(image, self._K, self._D, rvec, tvec, self._marker_size, 3)
            is_connected = True
            cv2.polylines(image, corner, is_connected, (0, 255, 0), 3)  # type: ignore[call-overload]
            cx = int((corner[0][0][0] + corner[0][2][0]) / 2)
            cy = int((corner[0][0][1] + corner[0][2][1]) / 2)
            cv2.putText(
                image,
                str(marker_id),
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3,
            )

        return image
