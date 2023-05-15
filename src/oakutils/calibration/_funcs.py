from typing import Tuple

import depthai as dai
import numpy as np
import cv2
import open3d as o3d

from ._classes import (
    CalibrationData,
    ColorCalibrationData,
    MonoCalibrationData,
    StereoCalibrationData,
)


def get_camera_calibration_basic(
    rgb_size: Tuple[int, int] = (1920, 1080), mono_size: Tuple[int, int] = (640, 400)
) -> CalibrationData:
    """
    Requires available OAK device.
    Get camera calibration data from OAK-D device.

    Parameters
    ----------
    rgb_size : Tuple[int, int], optional
        RGB camera resolution.
    mono_size : Tuple[int, int], optional
        Mono camera resolution.

    Returns
    -------
    CalibrationData
        Object containing all the calibration data.
    """
    with dai.Device() as device:
        calib_data = device.readCalibration2()

        K_rgb = np.array(
            calib_data.getCameraIntrinsics(
                dai.CameraBoardSocket.RGB, rgb_size[0], rgb_size[1]
            )
        )
        D_rgb = np.array(
            calib_data.getDistortionCoefficients(dai.CameraBoardSocket.RGB)
        )
        fx_rgb = K_rgb[0][0]
        fy_rgb = K_rgb[1][1]
        cx_rgb = K_rgb[0][2]
        cy_rgb = K_rgb[1][2]

        K_left = np.array(
            calib_data.getCameraIntrinsics(
                dai.CameraBoardSocket.LEFT, mono_size[0], mono_size[1]
            )
        )
        fx_left = K_left[0][0]
        fy_left = K_left[1][1]
        cx_left = K_left[0][2]
        cy_left = K_left[1][2]
        K_right = np.array(
            calib_data.getCameraIntrinsics(
                dai.CameraBoardSocket.RIGHT, mono_size[0], mono_size[1]
            )
        )

        fx_right = K_right[0][0]
        fy_right = K_right[1][1]
        cx_right = K_right[0][2]
        cy_right = K_right[1][2]
        D_left = np.array(
            calib_data.getDistortionCoefficients(dai.CameraBoardSocket.LEFT)
        )
        D_right = np.array(
            calib_data.getDistortionCoefficients(dai.CameraBoardSocket.RIGHT)
        )

        rgb_fov = calib_data.getFov(dai.CameraBoardSocket.RGB)
        left_fov = calib_data.getFov(dai.CameraBoardSocket.LEFT)
        right_fov = calib_data.getFov(dai.CameraBoardSocket.RIGHT)

        R1 = np.array(calib_data.getStereoLeftRectificationRotation())
        R2 = np.array(calib_data.getStereoRightRectificationRotation())

        T1 = (
            np.array(
                calib_data.getCameraTranslationVector(
                    srcCamera=dai.CameraBoardSocket.LEFT,
                    dstCamera=dai.CameraBoardSocket.RIGHT,
                )
            )
            / 100
        )  # convert to meters
        T2 = (
            np.array(
                calib_data.getCameraTranslationVector(
                    srcCamera=dai.CameraBoardSocket.RIGHT,
                    dstCamera=dai.CameraBoardSocket.LEFT,
                )
            )
            / 100
        )  # convert to meters
        T_l_rgb = (
            np.array(
                calib_data.getCameraTranslationVector(
                    srcCamera=dai.CameraBoardSocket.LEFT,
                    dstCamera=dai.CameraBoardSocket.RGB,
                )
            )
            / 100
        )  # convert to meters
        T_r_rgb = (
            np.array(
                calib_data.getCameraTranslationVector(
                    srcCamera=dai.CameraBoardSocket.RIGHT,
                    dstCamera=dai.CameraBoardSocket.RGB,
                )
            )
            / 100
        )  # convert to meters
        T_rgb_l = (
            np.array(
                calib_data.getCameraTranslationVector(
                    srcCamera=dai.CameraBoardSocket.RGB,
                    dstCamera=dai.CameraBoardSocket.LEFT,
                )
            )
            / 100
        )  # convert to meters
        T_rgb_r = (
            np.array(
                calib_data.getCameraTranslationVector(
                    srcCamera=dai.CameraBoardSocket.RGB,
                    dstCamera=dai.CameraBoardSocket.RIGHT,
                )
            )
            / 100
        )  # convert to meters

        H_left = np.matmul(np.matmul(K_right, R1), np.linalg.inv(K_left))
        H_right = np.matmul(np.matmul(K_right, R1), np.linalg.inv(K_right))

        l2r_extrinsic = np.array(
            calib_data.getCameraExtrinsics(
                srcCamera=dai.CameraBoardSocket.LEFT,
                dstCamera=dai.CameraBoardSocket.RIGHT,
            )
        )
        r2l_extrinsic = np.array(
            calib_data.getCameraExtrinsics(
                srcCamera=dai.CameraBoardSocket.RIGHT,
                dstCamera=dai.CameraBoardSocket.LEFT,
            )
        )
        l2rgb_extrinsic = np.array(
            calib_data.getCameraExtrinsics(
                srcCamera=dai.CameraBoardSocket.LEFT,
                dstCamera=dai.CameraBoardSocket.RGB,
            )
        )
        r2rgb_extrinsic = np.array(
            calib_data.getCameraExtrinsics(
                srcCamera=dai.CameraBoardSocket.RIGHT,
                dstCamera=dai.CameraBoardSocket.RGB,
            )
        )
        rgb2l_extrinsic = np.array(
            calib_data.getCameraExtrinsics(
                srcCamera=dai.CameraBoardSocket.RGB,
                dstCamera=dai.CameraBoardSocket.LEFT,
            )
        )
        rgb2r_extrinsic = np.array(
            calib_data.getCameraExtrinsics(
                srcCamera=dai.CameraBoardSocket.RGB,
                dstCamera=dai.CameraBoardSocket.RIGHT,
            )
        )

        baseline = calib_data.getBaselineDistance() / 100  # in meters

        rgb_data = ColorCalibrationData(
            size=rgb_size,
            K=K_rgb,
            D=D_rgb,
            fx=fx_rgb,
            fy=fy_rgb,
            cx=cx_rgb,
            cy=cy_rgb,
            fov=rgb_fov,
        )
        left_data = MonoCalibrationData(
            size=mono_size,
            K=K_left,
            D=D_left,
            fx=fx_left,
            fy=fy_left,
            cx=cx_left,
            cy=cy_left,
            fov=left_fov,
            R=R1,
            T=T1,
            H=H_left,
        )
        right_data = MonoCalibrationData(
            size=mono_size,
            K=K_right,
            D=D_right,
            fx=fx_right,
            fy=fy_right,
            cx=cx_right,
            cy=cy_right,
            fov=right_fov,
            R=R2,
            T=T2,
            H=H_right,
        )
        stereo_data = StereoCalibrationData(
            left=left_data,
            right=right_data,
            R1=R1,
            R2=R2,
            T1=T1,
            T2=T2,
            H_left=H_left,
            H_right=H_right,
            l2r_extrinsic=l2r_extrinsic,
            r2l_extrinsic=r2l_extrinsic,
            Q_left=create_q_matrix(fx_left, fy_left, cx_left, cy_left, -1.0*baseline),
            Q_right=create_q_matrix(fx_right, fy_right, cx_right, cy_right, baseline),
            baseline=baseline,
        )
        data = CalibrationData(
            rgb=rgb_data,
            left=left_data,
            right=right_data,
            stereo=stereo_data,
            l2rgb_extrinsic=l2rgb_extrinsic,
            r2rgb_extrinsic=r2rgb_extrinsic,
            rgb2l_extrinsic=rgb2l_extrinsic,
            rgb2r_extrinsic=rgb2r_extrinsic,
            T_l_rgb=T_l_rgb,
            T_r_rgb=T_r_rgb,
            T_rgb_l=T_rgb_l,
            T_rgb_r=T_rgb_r,
        )
    return data


def get_camera_calibration_primary_mono(
    rgb_size: Tuple[int, int] = (1920, 1080),
    mono_size: Tuple[int, int] = (640, 400),
    is_primary_mono_left: bool = True,
) -> CalibrationData:
    """
    Requires available OAK device.
    Get the calibration data for both RGB and mono cameras, as well as produce the
    primary mono camera calibration data. The primary mono camera is the one that has
    the depth aligned to it. The other mono camera is the secondary mono camera.

    Parameters
    ----------
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
    # load the data from get_camera_calibration
    data: CalibrationData = get_camera_calibration_basic(
        rgb_size=rgb_size, mono_size=mono_size
    )

    K_primary = data.left.K if is_primary_mono_left else data.right.K
    D_primary = data.left.D if is_primary_mono_left else data.right.D
    fx_primary = data.left.fx if is_primary_mono_left else data.right.fx
    fy_primary = data.left.fy if is_primary_mono_left else data.right.fy
    cx_primary = data.left.cx if is_primary_mono_left else data.right.cx
    cy_primary = data.left.cy if is_primary_mono_left else data.right.cy
    R_primary = data.stereo.R1 if is_primary_mono_left else data.stereo.R2
    T_primary = data.stereo.T1 if is_primary_mono_left else data.stereo.T2
    primary_extrinsic = (
        data.stereo.l2r_extrinsic if is_primary_mono_left else data.stereo.r2l_extrinsic
    )

    # create the primary mono camera calibration data
    primary_mono_data = MonoCalibrationData(
        size=mono_size,
        K=K_primary,
        D=D_primary,
        fx=fx_primary,
        fy=fy_primary,
        cx=cx_primary,
        cy=cy_primary,
        fov=data.left.fov,
        R=R_primary,
        T=T_primary,
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
            fx_primary, fy_primary, cx_primary, cy_primary, data.stereo.baseline
        ),
    )
    new_data = CalibrationData(
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
    return new_data


def create_q_matrix(fx: float, fy: float, cx: float, cy: float, baseline: float):
    """
    Create Q matrix for stereo depth map.

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

    Notes
    -----
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
        ]
    )


def get_camera_calibration(
    rgb_size: Tuple[int, int], mono_size: Tuple[int, int], is_primary_mono_left: bool
) -> CalibrationData:
    """
    Creates the full CalibrationData object, including the primary mono camera calibration data
    and the optional calculated values for OpenCV compatibility.

    Parameters
    ----------
    rgb_size : Tuple[int, int]
        RGB camera resolution.
    mono_size : Tuple[int, int]
        Mono camera resolution.
    is_primary_mono_left : bool
        Whether the primary mono camera is the left or right mono camera.

    Returns
    -------
    CalibrationData
        Object containing all the calibration data.
    """
    # get the data from get_camera_calibration_primary_mono
    data: CalibrationData = get_camera_calibration_primary_mono(
        rgb_size=rgb_size,
        mono_size=mono_size,
        is_primary_mono_left=is_primary_mono_left,
    )
    # run cv2.getOptimalNewCameraMatrix for RGB cam
    P_rgb, valid_region_rgb = cv2.getOptimalNewCameraMatrix(
        data.rgb.K,
        data.rgb.D,
        rgb_size,
        1,
        rgb_size,
    )
    map_rgb_1, map_rgb_2 = cv2.initUndistortRectifyMap(
        data.rgb.K,
        data.rgb.D,
        None,
        P_rgb,
        rgb_size,
        cv2.CV_16SC2,
    )
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
        P=P_rgb,
        valid_region=valid_region_rgb,
        map_1=map_rgb_1,
        map_2=map_rgb_2,
        pinhole=pinhole_rgb,
    )

    # run stereoRectify and initUndistortRectifyMap for left and right mono cams
    (
        cv2_R1,
        cv2_R2,
        P1,
        P2,
        cv2_Q,
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
        cv2_R1,
        P1,
        mono_size,
        cv2.CV_16SC2,
    )
    map_right_1, map_right_2 = cv2.initUndistortRectifyMap(
        data.right.K,
        data.right.D,
        cv2_R2,
        P2,
        mono_size,
        cv2.CV_16SC2,
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
        R=data.left.R,
        T=data.left.T,
        H=data.left.H,
        valid_region=valid_region_left,
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
        R=data.right.R,
        T=data.right.T,
        H=data.right.H,
        valid_region=valid_region_right,
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
        R=data.primary.R,
        T=data.primary.T,
        H=data.primary.H,
        valid_region=valid_region_primary,
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
        cv2_Q=cv2_Q,
        cv2_R1=cv2_R1,
        cv2_R2=cv2_R2,
        P1=P1,
        P2=P2,
        valid_region_primary=valid_region_primary,
        pinhole_primary=pinhole_primary,
    )

    # create final CalibrationData object
    new_data = CalibrationData(
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

    return new_data
