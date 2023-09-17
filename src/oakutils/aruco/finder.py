from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from typing_extensions import Self

    from oakutils.calibration import ColorCalibrationData, MonoCalibrationData


class ArucoFinder:
    """Class for finding aruco markers in images and acquiring
    transformation matrices to them

    Attributes
    ----------
    calibration : ColorCalibrationData, MonoCalibrationData, None

    Methods
    -------
    find(image: np.ndarray, rectified: bool | None = None)
      Finds the aruco markers in the image
    """

    def __init__(
        self: Self,
        aruco_dist: int = cv2.aruco.DICT_4X4_100,
        marker_size: float = 0.05,
        calibration: ColorCalibrationData | MonoCalibrationData | None = None,
    ) -> None:
        """Initializes the ArucoFinder class

        Parameters
        ----------
        aruco_dist : int, optional
            The aruco dictionary to use for finding markers,
              by default cv2.aruco.DICT_4X4_100
        marker_size : float, optional
            The size of the markers in meters, by default 0.05
        calibration : ColorCalibrationData, MonoCalibrationData, optional
            The calibration data to use for finding the transformation matrix,
              by default None
            Will utilize an identity matrix if not provided
        """
        self._adict = cv2.aruco.getPredefinedDictionary(aruco_dist)
        self._marker_size = marker_size
        self._calibration = calibration
        if self._calibration is None:
            self._K = np.zeros((3, 3), dtype=np.float32)
            self._D = np.zeros((5, 1), dtype=np.float32)
        else:
            self._K = self._calibration.K
            self._D = self._calibration.D

    @property
    def calibration(self: Self) -> ColorCalibrationData | MonoCalibrationData | None:
        """The calibration data used by the ArucoFinder

        Returns
        -------
        ColorCalibrationData, MonoCalibrationData, None
            The calibration data used by the ArucoFinder
        """
        return self._calibration

    @calibration.setter
    def calibration(
        self: Self, calibration: ColorCalibrationData | MonoCalibrationData
    ) -> None:
        """Sets the calibration data used by the ArucoFinder

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
        rectified: bool | None = None,
    ) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
        """Finds the aruco markers in the image.
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
        list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]
            The list of aruco markers found in the image
            Each tuple contains the id, transformation matrix,
              rotation vector, and translation vector
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
        corners, ids, _ = cv2.aruco.detectMarkers(image, self._adict)
        ret_val = []
        for idx, corner in enumerate(corners):
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corner], self._marker_size, self._K, self._D
            )
            try:
                rvec = rvecs[0]
                tvec = tvecs[0]
            except IndexError:
                continue
            r_matrix, _ = cv2.Rodrigues(rvec)  # Get equivalent 3x3 rotation matrix
            t_vector = tvec.T  # Get translation as a 3x1 vector
            t_matrix = np.block([[r_matrix, t_vector], [np.zeros((1, 3)), 1]])

            ret_val.append((ids[idx][0], t_matrix, rvec, tvec))

        return ret_val
