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

import cv2  # type: ignore[import]
import depthai as dai
import numpy as np
import open3d as o3d  # type: ignore[import]

from ._classes import (
    CalibrationData,
    ColorCalibrationData,
    MonoCalibrationData,
    StereoCalibrationData,
)


def get_camera_calibration_basic(
    device: dai.Device | None = None,
    rgb_size: tuple[int, int] = (1920, 1080),
    mono_size: tuple[int, int] = (640, 400),
) -> CalibrationData:
    """
    Use to get camera calibration data from OAK-D device.

    Note
    ----
        Requires available OAK device.
        If device is not provided, dai.Device() will be used.

    Parameters
    ----------
    device : Optional[dai.Device], optional
        DepthAI device object.
    rgb_size : Tuple[int, int], optional
        RGB camera resolution.
    mono_size : Tuple[int, int], optional
        Mono camera resolution.

    Returns
    -------
    CalibrationData
        Object containing all the calibration data.
    """
    if device is None:
        device = dai.Device()
    with device:
        calib_data = device.readCalibration2()

        k_rgb = np.array(
            calib_data.getCameraIntrinsics(
                dai.CameraBoardSocket.RGB,
                rgb_size[0],
                rgb_size[1],
            ),
        )
        d_rgb = np.array(
            calib_data.getDistortionCoefficients(dai.CameraBoardSocket.RGB),
        )
        fx_rgb = k_rgb[0][0]
        fy_rgb = k_rgb[1][1]
        cx_rgb = k_rgb[0][2]
        cy_rgb = k_rgb[1][2]

        k_left = np.array(
            calib_data.getCameraIntrinsics(
                dai.CameraBoardSocket.LEFT,
                mono_size[0],
                mono_size[1],
            ),
        )
        fx_left = k_left[0][0]
        fy_left = k_left[1][1]
        cx_left = k_left[0][2]
        cy_left = k_left[1][2]
        k_right = np.array(
            calib_data.getCameraIntrinsics(
                dai.CameraBoardSocket.RIGHT,
                mono_size[0],
                mono_size[1],
            ),
        )

        fx_right = k_right[0][0]
        fy_right = k_right[1][1]
        cx_right = k_right[0][2]
        cy_right = k_right[1][2]
        d_left = np.array(
            calib_data.getDistortionCoefficients(dai.CameraBoardSocket.LEFT),
        )
        d_right = np.array(
            calib_data.getDistortionCoefficients(dai.CameraBoardSocket.RIGHT),
        )

        rgb_fov = calib_data.getFov(dai.CameraBoardSocket.RGB)
        rgb_fov_rad = np.deg2rad(rgb_fov)
        left_fov = calib_data.getFov(dai.CameraBoardSocket.LEFT)
        left_fov_rad = np.deg2rad(left_fov)
        right_fov = calib_data.getFov(dai.CameraBoardSocket.RIGHT)
        right_fov_rad = np.deg2rad(right_fov)

        r1 = np.array(calib_data.getStereoLeftRectificationRotation())
        r2 = np.array(calib_data.getStereoRightRectificationRotation())

        t1 = (
            np.array(
                calib_data.getCameraTranslationVector(
                    srcCamera=dai.CameraBoardSocket.LEFT,
                    dstCamera=dai.CameraBoardSocket.RIGHT,
                ),
            )
            / 100
        )  # convert to meters
        t2 = (
            np.array(
                calib_data.getCameraTranslationVector(
                    srcCamera=dai.CameraBoardSocket.RIGHT,
                    dstCamera=dai.CameraBoardSocket.LEFT,
                ),
            )
            / 100
        )  # convert to meters
        t_l_rgb = (
            np.array(
                calib_data.getCameraTranslationVector(
                    srcCamera=dai.CameraBoardSocket.LEFT,
                    dstCamera=dai.CameraBoardSocket.RGB,
                ),
            )
            / 100
        )  # convert to meters
        t_r_rgb = (
            np.array(
                calib_data.getCameraTranslationVector(
                    srcCamera=dai.CameraBoardSocket.RIGHT,
                    dstCamera=dai.CameraBoardSocket.RGB,
                ),
            )
            / 100
        )  # convert to meters
        t_rgb_l = (
            np.array(
                calib_data.getCameraTranslationVector(
                    srcCamera=dai.CameraBoardSocket.RGB,
                    dstCamera=dai.CameraBoardSocket.LEFT,
                ),
            )
            / 100
        )  # convert to meters
        t_rgb_r = (
            np.array(
                calib_data.getCameraTranslationVector(
                    srcCamera=dai.CameraBoardSocket.RGB,
                    dstCamera=dai.CameraBoardSocket.RIGHT,
                ),
            )
            / 100
        )  # convert to meters

        h_left = np.matmul(np.matmul(k_right, r1), np.linalg.inv(k_left))
        h_right = np.matmul(np.matmul(k_right, r1), np.linalg.inv(k_right))

        l2r_extrinsic = np.array(
            calib_data.getCameraExtrinsics(
                srcCamera=dai.CameraBoardSocket.LEFT,
                dstCamera=dai.CameraBoardSocket.RIGHT,
            ),
        )
        r2l_extrinsic = np.array(
            calib_data.getCameraExtrinsics(
                srcCamera=dai.CameraBoardSocket.RIGHT,
                dstCamera=dai.CameraBoardSocket.LEFT,
            ),
        )
        l2rgb_extrinsic = np.array(
            calib_data.getCameraExtrinsics(
                srcCamera=dai.CameraBoardSocket.LEFT,
                dstCamera=dai.CameraBoardSocket.RGB,
            ),
        )
        r2rgb_extrinsic = np.array(
            calib_data.getCameraExtrinsics(
                srcCamera=dai.CameraBoardSocket.RIGHT,
                dstCamera=dai.CameraBoardSocket.RGB,
            ),
        )
        rgb2l_extrinsic = np.array(
            calib_data.getCameraExtrinsics(
                srcCamera=dai.CameraBoardSocket.RGB,
                dstCamera=dai.CameraBoardSocket.LEFT,
            ),
        )
        rgb2r_extrinsic = np.array(
            calib_data.getCameraExtrinsics(
                srcCamera=dai.CameraBoardSocket.RGB,
                dstCamera=dai.CameraBoardSocket.RIGHT,
            ),
        )

        baseline = calib_data.getBaselineDistance() / 100  # in meters

        rgb_data = ColorCalibrationData(
            size=rgb_size,
            K=k_rgb,
            D=d_rgb,
            fx=fx_rgb,
            fy=fy_rgb,
            cx=cx_rgb,
            cy=cy_rgb,
            fov=rgb_fov,
            fov_rad=rgb_fov_rad,
        )
        left_data = MonoCalibrationData(
            size=mono_size,
            K=k_left,
            D=d_left,
            fx=fx_left,
            fy=fy_left,
            cx=cx_left,
            cy=cy_left,
            fov=left_fov,
            fov_rad=left_fov_rad,
            R=r1,
            T=t1,
            H=h_left,
        )
        right_data = MonoCalibrationData(
            size=mono_size,
            K=k_right,
            D=d_right,
            fx=fx_right,
            fy=fy_right,
            cx=cx_right,
            cy=cy_right,
            fov=right_fov,
            fov_rad=right_fov_rad,
            R=r2,
            T=t2,
            H=h_right,
        )
        stereo_data = StereoCalibrationData(
            left=left_data,
            right=right_data,
            R1=r1,
            R2=r2,
            T1=t1,
            T2=t2,
            H_left=h_left,
            H_right=h_right,
            l2r_extrinsic=l2r_extrinsic,
            r2l_extrinsic=r2l_extrinsic,
            Q_left=create_q_matrix(fx_left, fy_left, cx_left, cy_left, -1.0 * baseline),
            Q_right=create_q_matrix(fx_right, fy_right, cx_right, cy_right, baseline),
            baseline=baseline,
        )
        return CalibrationData(
            rgb=rgb_data,
            left=left_data,
            right=right_data,
            stereo=stereo_data,
            l2rgb_extrinsic=l2rgb_extrinsic,
            r2rgb_extrinsic=r2rgb_extrinsic,
            rgb2l_extrinsic=rgb2l_extrinsic,
            rgb2r_extrinsic=rgb2r_extrinsic,
            T_l_rgb=t_l_rgb,
            T_r_rgb=t_r_rgb,
            T_rgb_l=t_rgb_l,
            T_rgb_r=t_rgb_r,
        )


def get_camera_calibration_primary_mono(
    device: dai.Device | None = None,
    rgb_size: tuple[int, int] = (1920, 1080),
    mono_size: tuple[int, int] = (640, 400),
    *,
    is_primary_mono_left: bool | None = None,
) -> CalibrationData:
    """
    Use to get the calibration data for both RGB and mono cameras and primary mono camera.

    Note
    ----
        Requires available OAK device.

    Parameters
    ----------
    device : Optional[dai.Device], optional
        DepthAI device object.
    rgb_size : Tuple[int, int], optional
        RGB camera resolution.
    mono_size : Tuple[int, int], optional
        Mono camera resolution.
    is_primary_mono_left : bool, optional
        Whether the primary mono camera is the left or right mono camera.

    Returns
    -------
    CalibrationData
        Object containing all the calibration data.
    """
    if is_primary_mono_left is None:
        is_primary_mono_left = True
    # load the data from get_camera_calibration
    data: CalibrationData = get_camera_calibration_basic(
        device=device,
        rgb_size=rgb_size,
        mono_size=mono_size,
    )

    k_primary = data.left.K if is_primary_mono_left else data.right.K
    d_primary = data.left.D if is_primary_mono_left else data.right.D
    fx_primary = data.left.fx if is_primary_mono_left else data.right.fx
    fy_primary = data.left.fy if is_primary_mono_left else data.right.fy
    cx_primary = data.left.cx if is_primary_mono_left else data.right.cx
    cy_primary = data.left.cy if is_primary_mono_left else data.right.cy
    fov_primary = data.left.fov if is_primary_mono_left else data.right.fov
    fov_rad_primary = data.left.fov_rad if is_primary_mono_left else data.right.fov_rad
    r_primary = data.stereo.R1 if is_primary_mono_left else data.stereo.R2
    t_primary = data.stereo.T1 if is_primary_mono_left else data.stereo.T2
    primary_extrinsic = (
        data.stereo.l2r_extrinsic if is_primary_mono_left else data.stereo.r2l_extrinsic
    )

    # create the primary mono camera calibration data
    primary_mono_data = MonoCalibrationData(
        size=mono_size,
        K=k_primary,
        D=d_primary,
        fx=fx_primary,
        fy=fy_primary,
        cx=cx_primary,
        cy=cy_primary,
        fov=fov_primary,
        fov_rad=fov_rad_primary,
        R=r_primary,
        T=t_primary,
        H=primary_extrinsic,
    )

    # return all the data, including from get_camera_calibration
    stereo = StereoCalibrationData(
        left=data.stereo.left,
        right=data.stereo.right,
        R1=data.stereo.R1,
        R2=data.stereo.R2,
        T1=data.stereo.T1,
        T2=data.stereo.T2,
        H_left=data.stereo.H_left,
        H_right=data.stereo.H_right,
        l2r_extrinsic=data.stereo.l2r_extrinsic,
        r2l_extrinsic=data.stereo.r2l_extrinsic,
        Q_left=data.stereo.Q_left,
        Q_right=data.stereo.Q_right,
        baseline=data.stereo.baseline,
        primary=primary_mono_data,
        Q_primary=create_q_matrix(
            fx_primary,
            fy_primary,
            cx_primary,
            cy_primary,
            data.stereo.baseline,
        ),
    )
    return CalibrationData(
        left=data.left,
        right=data.right,
        rgb=data.rgb,
        stereo=stereo,
        l2rgb_extrinsic=data.l2rgb_extrinsic,
        r2rgb_extrinsic=data.r2rgb_extrinsic,
        rgb2l_extrinsic=data.rgb2l_extrinsic,
        rgb2r_extrinsic=data.rgb2r_extrinsic,
        T_l_rgb=data.T_l_rgb,
        T_r_rgb=data.T_r_rgb,
        T_rgb_l=data.T_rgb_l,
        T_rgb_r=data.T_rgb_r,
        primary=primary_mono_data,
    )


def create_q_matrix(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    baseline: float,
) -> np.ndarray:
    """
    Use to create Q matrix for stereo depth map.

    Parameters
    ----------
    fx : float
        Focal length in x direction (in millimeters).
    fy : float
        Focal length in y direction (in millimeters).
    cx : float
        Principal point in x direction.
    cy : float
        Principal point in y direction.
    baseline : float
        Baseline distance between the left and right camera (in meters).

    Returns
    -------
    np.ndarray
        Q matrix for stereo depth map.

    Note
    ----
        This uses the OpenCV formula for Q matrix, with an alpha value of 0.
        Thus, the Q matrix is:
        [[1, 0, 0, -cx],
        [0, 1, 0, -cy],
        [0, 0, 0, (fx + fy) / 2],
        [0, 0, -1 / baseline, 0]]
    """
    return np.array(
        [
            [1, 0, 0, -cx],
            [0, 1, 0, -cy],
            [0, 0, 0, (fx + fy) / 2],
            [0, 0, -1.0 / baseline, 0],
        ],
    )


def get_camera_calibration(
    rgb_size: tuple[int, int],
    mono_size: tuple[int, int],
    device: dai.Device | None = None,
    *,
    is_primary_mono_left: bool | None = None,
) -> CalibrationData:
    """
    Use to create the full CalibrationData object.

    This includes the calibration data for the RGB camera, the left mono camera,
    the right mono camera, and the primary mono camera.
    As well as all OpenCV and Open3D compatible data for each camera.


    Parameters
    ----------
    rgb_size : Tuple[int, int]
        RGB camera resolution.
    mono_size : Tuple[int, int]
        Mono camera resolution.
    is_primary_mono_left : Optional[bool], optional
        Whether the primary mono camera is the left or right mono camera.
        Defaults to True.
    device : Optional[dai.Device], optional
        DepthAI device object.

    Returns
    -------
    CalibrationData
        Object containing all the calibration data.
    """
    if is_primary_mono_left is None:
        is_primary_mono_left = True
    # get the data from get_camera_calibration_primary_mono
    data: CalibrationData = get_camera_calibration_primary_mono(
        device=device,
        rgb_size=rgb_size,
        mono_size=mono_size,
        is_primary_mono_left=is_primary_mono_left,
    )
    # assert data.primary is not None  # help mypy
    if data.primary is None:
        err_msg = "data.primary is None"
        raise RuntimeError(err_msg)

    q_primary = create_q_matrix(
        data.primary.fx,
        data.primary.fy,
        data.primary.cx,
        data.primary.cy,
        data.stereo.baseline,
    )

    # run cv2.getOptimalNewCameraMatrix for RGB cam
    p_rgb, valid_region_rgb = cv2.getOptimalNewCameraMatrix(
        data.rgb.K,
        data.rgb.D,
        rgb_size,
        1,
        rgb_size,
    )
    map_rgb_1, map_rgb_2 = cv2.initUndistortRectifyMap(
        data.rgb.K,
        data.rgb.D,
        None,  # pyright: ignore[reportArgumentType]
        p_rgb,
        rgb_size,
        cv2.CV_16SC2,  # type: ignore[attr-defined]
    )  # type: ignore[call-overload]
    pinhole_rgb = o3d.camera.PinholeCameraIntrinsic(
        width=rgb_size[0],
        height=rgb_size[1],
        fx=data.rgb.fx,
        fy=data.rgb.fy,
        cx=data.rgb.cx,
        cy=data.rgb.cy,
    )

    # add the pinhole data and maps to the data.rgb object
    rgb = ColorCalibrationData(
        size=data.rgb.size,
        K=data.rgb.K,
        D=data.rgb.D,
        fx=data.rgb.fx,
        fy=data.rgb.fy,
        cx=data.rgb.cx,
        cy=data.rgb.cy,
        fov=data.rgb.fov,
        fov_rad=data.rgb.fov_rad,
        P=p_rgb,
        valid_region=valid_region_rgb,  # type: ignore[arg-type]
        map_1=map_rgb_1,
        map_2=map_rgb_2,
        pinhole=pinhole_rgb,
    )

    # run stereoRectify and initUndistortRectifyMap for left and right mono cams
    (
        cv2_r1,
        cv2_r2,
        p1,
        p2,
        cv2_q,
        valid_region_left,
        valid_region_right,
    ) = cv2.stereoRectify(
        data.left.K,
        data.left.D,
        data.right.K,
        data.right.D,
        mono_size,
        data.primary.R,
        data.primary.T,
    )
    map_left_1, map_left_2 = cv2.initUndistortRectifyMap(
        data.left.K,
        data.left.D,
        cv2_r1,
        p1,
        mono_size,
        cv2.CV_16SC2,  # type: ignore[attr-defined]
    )
    map_right_1, map_right_2 = cv2.initUndistortRectifyMap(
        data.right.K,
        data.right.D,
        cv2_r2,
        p2,
        mono_size,
        cv2.CV_16SC2,  # type: ignore[attr-defined]
    )
    valid_region_primary = (
        valid_region_left if is_primary_mono_left else valid_region_right
    )
    map_1_primary = map_left_1 if is_primary_mono_left else map_right_1
    map_2_primary = map_left_2 if is_primary_mono_left else map_right_2

    # create o3d PinholeCameraIntrinsic objects for left, right, and primary mono cams
    pinhole_left = o3d.camera.PinholeCameraIntrinsic(
        width=mono_size[0],
        height=mono_size[1],
        fx=data.left.fx,
        fy=data.left.fy,
        cx=data.left.cx,
        cy=data.left.cy,
    )
    pinhole_right = o3d.camera.PinholeCameraIntrinsic(
        width=mono_size[0],
        height=mono_size[1],
        fx=data.right.fx,
        fy=data.right.fy,
        cx=data.right.cx,
        cy=data.right.cy,
    )
    pinhole_primary = pinhole_left if is_primary_mono_left else pinhole_right

    # add the data to the data.left, data.right, and data.primary objects
    left = MonoCalibrationData(
        size=data.left.size,
        K=data.left.K,
        D=data.left.D,
        fx=data.left.fx,
        fy=data.left.fy,
        cx=data.left.cx,
        cy=data.left.cy,
        fov=data.left.fov,
        fov_rad=data.left.fov_rad,
        R=data.left.R,
        T=data.left.T,
        H=data.left.H,
        valid_region=valid_region_left,  # type: ignore[arg-type]
        map_1=map_left_1,
        map_2=map_left_2,
        pinhole=pinhole_left,
    )
    right = MonoCalibrationData(
        size=data.right.size,
        K=data.right.K,
        D=data.right.D,
        fx=data.right.fx,
        fy=data.right.fy,
        cx=data.right.cx,
        cy=data.right.cy,
        fov=data.right.fov,
        fov_rad=data.right.fov_rad,
        R=data.right.R,
        T=data.right.T,
        H=data.right.H,
        valid_region=valid_region_right,  # type: ignore[arg-type]
        map_1=map_right_1,
        map_2=map_right_2,
        pinhole=pinhole_right,
    )
    primary = MonoCalibrationData(
        size=data.primary.size,
        K=data.primary.K,
        D=data.primary.D,
        fx=data.primary.fx,
        fy=data.primary.fy,
        cx=data.primary.cx,
        cy=data.primary.cy,
        fov=data.primary.fov,
        fov_rad=data.primary.fov_rad,
        R=data.primary.R,
        T=data.primary.T,
        H=data.primary.H,
        valid_region=valid_region_primary,  # type: ignore[arg-type]
        map_1=map_1_primary,
        map_2=map_2_primary,
        pinhole=pinhole_primary,
    )

    # add the data to the data.stereo object
    stereo = StereoCalibrationData(
        left=data.stereo.left,
        right=data.stereo.right,
        R1=data.stereo.R1,
        R2=data.stereo.R2,
        T1=data.stereo.T1,
        T2=data.stereo.T2,
        H_left=data.stereo.H_left,
        H_right=data.stereo.H_right,
        l2r_extrinsic=data.stereo.l2r_extrinsic,
        r2l_extrinsic=data.stereo.r2l_extrinsic,
        Q_left=data.stereo.Q_left,
        Q_right=data.stereo.Q_right,
        baseline=data.stereo.baseline,
        primary=primary,
        Q_primary=q_primary,
        Q_cv2=cv2_q,
        R1_cv2=cv2_r1,
        R2_cv2=cv2_r2,
        P1=p1,
        P2=p2,
        valid_region_primary=valid_region_primary,  # type: ignore[arg-type]
        pinhole_primary=pinhole_primary,
    )

    # create final CalibrationData object
    return CalibrationData(
        rgb=rgb,
        left=left,
        right=right,
        stereo=stereo,
        l2rgb_extrinsic=data.l2rgb_extrinsic,
        r2rgb_extrinsic=data.r2rgb_extrinsic,
        rgb2l_extrinsic=data.rgb2l_extrinsic,
        rgb2r_extrinsic=data.rgb2r_extrinsic,
        T_l_rgb=data.T_l_rgb,
        T_r_rgb=data.T_r_rgb,
        T_rgb_l=data.T_rgb_l,
        T_rgb_r=data.T_rgb_r,
        primary=primary,
    )
