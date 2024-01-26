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
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import open3d as o3d  # type: ignore[import]


@dataclass(frozen=True)
class MonoCalibrationData:
    """
    Class to store calibration data for a mono camera.

    Attributes
    ----------
    size : Tuple[int, int]
        Image size.
    K : np.ndarray
        Camera matrix.
    D : np.ndarray
        Distortion coefficients.
    fx : float
        Focal length in x.
    fy : float
        Focal length in y.
    cx : float
        Principal point in x.
    cy : float
        Principal point in y.
    fov : float
        Field of view.
    fov_rad : float
        Field of view in radians.
    R : np.ndarray
        Rotation matrix.
    T : np.ndarray
        Translation matrix.
    H : np.ndarray
        Homography matrix.
    valid_region : Optional[Tuple[int, int, int, int]], optional
        Valid region of the calibration generated by cv2.stereoRectify.
    map_1 : Optional[np.ndarray], optional
        Map 1 for undistortion generated by cv2.initUndistortRectifyMap.
    map_2 : Optional[np.ndarray], optional
        Map 2 for undistortion generated by cv2.initUndistortRectifyMap.
    pinhole : Optional[o3d.camera.PinholeCameraIntrinsic], optional
        o3d pinhole camera intrinsic.
    """

    size: tuple[int, int]
    K: np.ndarray
    D: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    fov: float
    fov_rad: float
    R: np.ndarray
    T: np.ndarray
    H: np.ndarray
    valid_region: tuple[int, int, int, int] | None = None
    map_1: np.ndarray | None = None
    map_2: np.ndarray | None = None
    pinhole: o3d.camera.PinholeCameraIntrinsic | None = None


@dataclass(frozen=True)
class StereoCalibrationData:
    """
    Class to store calibration data for stereo cameras.

    Attributes
    ----------
    left : MonoCalibrationData
        Left camera calibration data.
    right : MonoCalibrationData
        Right camera calibration data.
    R1 : np.ndarray
        Rectification transform for the left camera.
    R2 : np.ndarray
        Rectification transform for the right camera.
    T1 : np.ndarray
        Projection matrix for the left camera.
    T2 : np.ndarray
        Projection matrix for the right camera.
    H_left : np.ndarray
        Homography matrix for the left camera.
    H_right : np.ndarray
        Homography matrix for the right camera.
    l2r_extrinsic : np.ndarray
        Extrinsic matrix from the left to the right camera.
    r2l_extrinsic : np.ndarray
        Extrinsic matrix from the right to the left camera.
    Q_left : np.ndarray
        Q matrix for the left camera.
    Q_right : np.ndarray
        Q matrix for the right camera.
    baseline : float
        Baseline between the two cameras (in meters).
    primary : Optional[MonoCalibrationData], optional
        Primary camera calibration data.
    Q_primary : Optional[np.ndarray], optional
        Q matrix for the primary camera.
    Q_cv2 : Optional[np.ndarray], optional
        Q matrix generated by cv2.stereoRectify.
    R1_cv2 : Optional[np.ndarray], optional
        R1 matrix generated by cv2.stereoRectify.
    R2_cv2 : Optional[np.ndarray], optional
        R2 matrix generated by cv2.stereoRectify.
    P1 : Optional[np.ndarray], optional
        P1 matrix generated by cv2.stereoRectify.
    P2 : Optional[np.ndarray], optional
        P2 matrix generated by cv2.stereoRectify.
    valid_region_primary : Optional[Tuple[int, int, int, int]], optional
        Valid region of the primary camera.
    pinhole_primary : Optional[o3d.camera.PinholeCameraIntrinsic], optional
        o3d pinhole camera intrinsic for the primary camera.

    See Also
    --------
    MonoCalibrationData : Class to store calibration data for a mono camera.
    """

    left: MonoCalibrationData
    right: MonoCalibrationData
    R1: np.ndarray
    R2: np.ndarray
    T1: np.ndarray
    T2: np.ndarray
    H_left: np.ndarray
    H_right: np.ndarray
    l2r_extrinsic: np.ndarray
    r2l_extrinsic: np.ndarray
    Q_left: np.ndarray
    Q_right: np.ndarray
    baseline: float
    primary: MonoCalibrationData | None = None
    Q_primary: np.ndarray | None = None
    Q_cv2: np.ndarray | None = None
    R1_cv2: np.ndarray | None = None
    R2_cv2: np.ndarray | None = None
    P1: np.ndarray | None = None
    P2: np.ndarray | None = None
    valid_region_primary: tuple[int, int, int, int] | None = None
    pinhole_primary: o3d.camera.PinholeCameraIntrinsic | None = None


@dataclass(frozen=True)
class ColorCalibrationData:
    """
    Class to store calibration data for a color camera.

    Attributes
    ----------
    size : Tuple[int, int]
        Image size.
    K : np.ndarray
        Camera matrix.
    D : np.ndarray
        Distortion coefficients.
    fx : float
        Focal length in the x direction.
    fy : float
        Focal length in the y direction.
    cx : float
        Principal point in the x direction.
    cy : float
        Principal point in the y direction.
    fov : float
        Field of view.
    fov_rad : float
        Field of view in radians.
    P : Optional[np.ndarray], optional
        Projection matrix.
    valid_region : Optional[Tuple[int, int, int, int]], optional
        Valid region of the calibration generated by cv2.getOptimalNewCameraMatrix.
    map_1 : Optional[np.ndarray], optional
        Map 1 for undistortion generated by cv2.initUndistortRectifyMap.
    map_2 : Optional[np.ndarray], optional
        Map 2 for undistortion generated by cv2.initUndistortRectifyMap.
    pinhole : Optional[o3d.camera.PinholeCameraIntrinsic], optional
        o3d pinhole camera intrinsic.
    """

    size: tuple[int, int]
    K: np.ndarray
    D: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    fov: float
    fov_rad: float
    P: np.ndarray | None = None
    valid_region: tuple[int, int, int, int] | None = None
    map_1: np.ndarray | None = None
    map_2: np.ndarray | None = None
    pinhole: o3d.camera.PinholeCameraIntrinsic | None = None


@dataclass(frozen=True)
class CalibrationData:
    """
    An object to store calibration data for an entire OAK camera.

    Attributes
    ----------
    rgb : ColorCalibrationData
        RGB camera calibration data.
    left : MonoCalibrationData
        Left mono camera calibration data.
    right : MonoCalibrationData
        Right mono camera calibration data.
    stereo : StereoCalibrationData
        Stereo camera calibration data.
    l2rgb_extrinsic : np.ndarray
        Extrinsic matrix from the left to the RGB camera.
    r2rgb_extrinsic : np.ndarray
        Extrinsic matrix from the right to the RGB camera.
    rgb2l_extrinsic : np.ndarray
        Extrinsic matrix from the RGB to the left camera.
    rgb2r_extrinsic : np.ndarray
        Extrinsic matrix from the RGB to the right camera.
    T_l_rgb : np.ndarray
        Translation vector from the left to the RGB camera.
    T_r_rgb : np.ndarray
        Translation vector from the right to the RGB camera.
    T_rgb_l : np.ndarray
        Translation vector from the RGB to the left camera.
    T_rgb_r : np.ndarray
        Translation vector from the RGB to the right camera.
    primary : Optional[MonoCalibrationData], optional
        Primary camera calibration data.

    See Also
    --------
    ColorCalibrationData : Class to store calibration data for a color camera.
    MonoCalibrationData : Class to store calibration data for a mono camera.
    StereoCalibrationData : Class to store calibration data for stereo cameras.
    """

    rgb: ColorCalibrationData
    left: MonoCalibrationData
    right: MonoCalibrationData
    stereo: StereoCalibrationData
    l2rgb_extrinsic: np.ndarray
    r2rgb_extrinsic: np.ndarray
    rgb2l_extrinsic: np.ndarray
    rgb2r_extrinsic: np.ndarray
    T_l_rgb: np.ndarray
    T_r_rgb: np.ndarray
    T_rgb_l: np.ndarray
    T_rgb_r: np.ndarray
    primary: MonoCalibrationData | None = None
