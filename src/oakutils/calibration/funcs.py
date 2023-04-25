from typing import Tuple

import depthai as dai
import numpy as np
import cv2
import open3d as o3d

from .classes import (
    CalibrationData,
    ColorCalibrationData,
    MonoCalibrationData,
    StereoCalibrationData,
)


def get_camera_calibration(
    rgb_size: Tuple[int, int] = (1920, 1080), mono_size: Tuple[int, int] = (640, 400)
) -> Tuple[
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    """
    Requires available OAK device.
    Get camera calibration data from OAK-D device.
    Params:
        rgb_size: Tuple[int, int] = (1920, 1080)
            RGB camera resolution.
        mono_size: Tuple[int, int] = (640, 400)
            Mono camera resolution.
    Returns:
        K_rgb: np.ndarray
            RGB camera intrinsic matrix.
        D_rgb: np.ndarray
            RGB camera distortion coefficients.
        fx_rgb: float
            RGB camera focal length in x direction. (in millimeters)
        fy_rgb: float
            RGB camera focal length in y direction. (in millimeters)
        cx_rgb: float
            RGB camera principal point in x direction.
        cy_rgb: float
            RGB camera principal point in y direction.
        K_left: np.ndarray
            Left mono camera intrinsic matrix.
        D_left: np.ndarray
            Left mono camera distortion coefficients.
        fx_left: float
            Left mono camera focal length in x direction. (in millimeters)
        fy_left: float
            Left mono camera focal length in y direction. (in millimeters)
        cx_left: float
            Left mono camera principal point in x direction.
        cy_left: float
            Left mono camera principal point in y direction.
        K_right: np.ndarray
            Right mono camera intrinsic matrix.
        D_right: np.ndarray
            Right mono camera distortion coefficients.
        fx_right: float
            Right mono camera focal length in x direction. (in millimeters)
        fy_right: float
            Right mono camera focal length in y direction. (in millimeters)
        cx_right: float
            Right mono camera principal point in x direction.
        cy_right: float
            Right mono camera principal point in y direction.
        rgb_fov: float
            RGB camera field of view. (in degrees)
        mono_fov: float
            Mono camera field of view. (in degrees)
        R1: np.ndarray
            Rectification transform (rotation matrix) for the left camera.
        R2: np.ndarray
            Rectification transform (rotation matrix) for the right camera.
        T1: np.ndarray
            Translation vector for the left camera. (in meters)
        T2: np.ndarray
            Translation vector for the right camera. (in meters)
        H_left: np.ndarray
            Rectification homography matrix for the left camera.
        H_right: np.ndarray
            Rectification homography matrix for the right camera.
        l2r_extrinsic: np.ndarray
            Extrinsic matrix from left to right camera.
        r2l_extrinsic: np.ndarray
            Extrinsic matrix from right to left camera.
        baseline: float
            Baseline between left and right camera. (in meters)
    """
    with dai.Device() as device:
        calibData = device.readCalibration2()

        K_rgb = np.array(
            calibData.getCameraIntrinsics(
                dai.CameraBoardSocket.RGB, rgb_size[0], rgb_size[1]
            )
        )
        D_rgb = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.RGB))
        fx_rgb = K_rgb[0][0]
        fy_rgb = K_rgb[1][1]
        cx_rgb = K_rgb[0][2]
        cy_rgb = K_rgb[1][2]

        K_left = np.array(
            calibData.getCameraIntrinsics(
                dai.CameraBoardSocket.LEFT, mono_size[0], mono_size[1]
            )
        )
        fx_left = K_left[0][0]
        fy_left = K_left[1][1]
        cx_left = K_left[0][2]
        cy_left = K_left[1][2]
        K_right = np.array(
            calibData.getCameraIntrinsics(
                dai.CameraBoardSocket.RIGHT, mono_size[0], mono_size[1]
            )
        )

        fx_right = K_right[0][0]
        fy_right = K_right[1][1]
        cx_right = K_right[0][2]
        cy_right = K_right[1][2]
        D_left = np.array(
            calibData.getDistortionCoefficients(dai.CameraBoardSocket.LEFT)
        )
        D_right = np.array(
            calibData.getDistortionCoefficients(dai.CameraBoardSocket.RIGHT)
        )

        rgb_fov = calibData.getFov(dai.CameraBoardSocket.RGB)
        mono_fov = calibData.getFov(dai.CameraBoardSocket.LEFT)

        R1 = np.array(calibData.getStereoLeftRectificationRotation())
        R2 = np.array(calibData.getStereoRightRectificationRotation())

        T1 = (
            np.array(
                calibData.getCameraTranslationVector(
                    srcCamera=dai.CameraBoardSocket.LEFT,
                    dstCamera=dai.CameraBoardSocket.RIGHT,
                )
            )
            / 100
        )  # convert to meters
        T2 = (
            np.array(
                calibData.getCameraTranslationVector(
                    srcCamera=dai.CameraBoardSocket.RIGHT,
                    dstCamera=dai.CameraBoardSocket.LEFT,
                )
            )
            / 100
        )  # convert to meters

        H_left = np.matmul(np.matmul(K_right, R1), np.linalg.inv(K_left))
        H_right = np.matmul(np.matmul(K_right, R1), np.linalg.inv(K_right))

        l2r_extrinsic = np.array(
            calibData.getCameraExtrinsics(
                srcCamera=dai.CameraBoardSocket.LEFT,
                dstCamera=dai.CameraBoardSocket.RIGHT,
            )
        )
        r2l_extrinsic = np.array(
            calibData.getCameraExtrinsics(
                srcCamera=dai.CameraBoardSocket.RIGHT,
                dstCamera=dai.CameraBoardSocket.LEFT,
            )
        )

        baseline = calibData.getBaselineDistance() / 100  # in meters

        return (
            K_rgb,
            D_rgb,
            fx_rgb,
            fy_rgb,
            cx_rgb,
            cy_rgb,
            K_left,
            D_left,
            fx_left,
            fy_left,
            cx_left,
            cy_left,
            K_right,
            D_right,
            fx_right,
            fy_right,
            cx_right,
            cy_right,
            rgb_fov,
            mono_fov,
            R1,
            R2,
            T1,
            T2,
            H_left,
            H_right,
            l2r_extrinsic,
            r2l_extrinsic,
            baseline,
        )


def get_camera_calibration_primary_mono(
    rgb_size: Tuple[int, int] = (1920, 1080),
    mono_size: Tuple[int, int] = (640, 400),
    is_primary_mono_left: bool = True,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Requires available OAK device.
    Get the calibration data for both rgb and mono cameras, as well as produces the
    primary mono camera calibration data. The primary mono camera is the one that has
    the depth aligned to it. The other mono camera is the secondary mono camera.
    Params:
        rgb_size: Tuple[int, int] = (1920, 1080)
            RGB camera resolution.
        mono_size: Tuple[int, int] = (640, 400)
            Mono camera resolution.
        is_primary_mono_left: bool = True
            Whether the primary mono camera is the left or right mono camera.
    Returns:
        K_rgb: np.ndarray
            RGB camera intrinsic matrix.
        D_rgb: np.ndarray
            RGB camera distortion coefficients.
        fx_rgb: float
            RGB camera focal length in x direction. (in millimeters)
        fy_rgb: float
            RGB camera focal length in y direction. (in millimeters)
        cx_rgb: float
            RGB camera principal point in x direction.
        cy_rgb: float
            RGB camera principal point in y direction.
        K_left: np.ndarray
            Left mono camera intrinsic matrix.
        D_left: np.ndarray
            Left mono camera distortion coefficients.
        fx_left: float
            Left mono camera focal length in x direction. (in millimeters)
        fy_left: float
            Left mono camera focal length in y direction. (in millimeters)
        cx_left: float
            Left mono camera principal point in x direction.
        cy_left: float
            Left mono camera principal point in y direction.
        K_right: np.ndarray
            Right mono camera intrinsic matrix.
        D_right: np.ndarray
            Right mono camera distortion coefficients.
        fx_right: float
            Right mono camera focal length in x direction. (in millimeters)
        fy_right: float
            Right mono camera focal length in y direction. (in millimeters)
        cx_right: float
            Right mono camera principal point in x direction.
        cy_right: float
            Right mono camera principal point in y direction.
        rgb_fov: float
            RGB camera field of view. (in degrees)
        mono_fov: float
            Mono camera field of view. (in degrees)
        R1: np.ndarray
            Rectification transform (rotation matrix) for the left camera.
        R2: np.ndarray
            Rectification transform (rotation matrix) for the right camera.
        T1: np.ndarray
            Translation vector for the left camera. (in meters)
        T2: np.ndarray
            Translation vector for the right camera. (in meters)
        H_left: np.ndarray
            Rectification homography matrix for the left camera.
        H_right: np.ndarray
            Rectification homography matrix for the right camera.
        l2r_extrinsic: np.ndarray
            Extrinsic matrix from left to right camera.
        r2l_extrinsic: np.ndarray
            Extrinsic matrix from right to left camera.
        baseline: float
            Baseline between left and right camera. (in meters)
        K_primary: np.ndarray
            Primary mono camera intrinsic matrix.
        D_primary: np.ndarray
            Primary mono camera distortion coefficients.
        fx_primary: float
            Primary mono camera focal length in x direction. (in millimeters)
        fy_primary: float
            Primary mono camera focal length in y direction. (in millimeters)
        cx_primary: float
            Primary mono camera principal point in x direction.
        cy_primary: float
            Primary mono camera principal point in y direction.
        R_primary: np.ndarray
            Rectification transform (rotation matrix) for the primary mono camera.
        T_primary: np.ndarray
            Translation vector for the primary mono camera. (in meters)
        primary_extrinsic: np.ndarray
            Extrinsic matrix from primary mono camera to the secondary mono camera.
    """
    # load the data from get_camera_calibration
    (
        K_rgb,
        D_rgb,
        fx_rgb,
        fy_rgb,
        cx_rgb,
        cy_rgb,
        K_left,
        D_left,
        fx_left,
        fy_left,
        cx_left,
        cy_left,
        K_right,
        D_right,
        fx_right,
        fy_right,
        cx_right,
        cy_right,
        rgb_fov,
        mono_fov,
        R1,
        R2,
        T1,
        T2,
        H_left,
        H_right,
        l2r_extrinsic,
        r2l_extrinsic,
        baseline,
    ) = get_camera_calibration(rgb_size=rgb_size, mono_size=mono_size)

    K_primary = K_left if is_primary_mono_left else K_right
    D_primary = D_left if is_primary_mono_left else D_right
    fx_primary = fx_left if is_primary_mono_left else fx_right
    fy_primary = fy_left if is_primary_mono_left else fy_right
    cx_primary = cx_left if is_primary_mono_left else cx_right
    cy_primary = cy_left if is_primary_mono_left else cy_right
    R_primary = R1 if is_primary_mono_left else R2
    T_primary = T1 if is_primary_mono_left else T2
    primary_extrinsic = l2r_extrinsic if is_primary_mono_left else r2l_extrinsic

    # return all the data, including from get_camera_calibration
    return (
        K_rgb,
        D_rgb,
        fx_rgb,
        fy_rgb,
        cx_rgb,
        cy_rgb,
        K_left,
        D_left,
        fx_left,
        fy_left,
        cx_left,
        cy_left,
        K_right,
        D_right,
        fx_right,
        fy_right,
        cx_right,
        cy_right,
        rgb_fov,
        mono_fov,
        R1,
        R2,
        T1,
        T2,
        H_left,
        H_right,
        l2r_extrinsic,
        r2l_extrinsic,
        baseline,
        K_primary,
        D_primary,
        fx_primary,
        fy_primary,
        cx_primary,
        cy_primary,
        R_primary,
        T_primary,
        primary_extrinsic,
    )


def create_q_matrix(fx: float, fy: float, cx: float, cy: float, baseline: float):
    """
    Create Q matrix for stereo depth map.
    Params:
        fx: float
            Focal length in x direction. (in millimeters)
        fy: float
            Focal length in y direction. (in millimeters)
        cx: float
            Principal point in x direction.
        cy: float
            Principal point in y direction.
        baseline: float
            Baseline distance between left and right camera. (in meters)
    """
    return np.array(
        [
            1,
            0,
            0,
            -cx,
            0,
            1,
            0,
            -cy,
            0,
            0,
            0,
            (fx + fy) // 2,
            0,
            0,
            -1 / baseline,
            (cx - cy) / baseline,
        ]
    ).reshape(4, 4)


def create_camera_calibration(
    rgb_size: Tuple[int, int], mono_size: Tuple[int, int], is_primary_mono_left: bool
) -> CalibrationData:
    """
    Wrapper around 'get_camera_calibration_primary_mono' to create a CalibrationData object
    Params:
        rgb_size: Tuple[int, int] = (1920, 1080)
            RGB camera resolution.
        mono_size: Tuple[int, int] = (640, 400)
            Mono camera resolution.
        is_primary_mono_left: bool = True
            Whether the primary mono camera is the left or right mono camera.
    Returns:
        CalibrationData
            Object containing all the calibration data.
    """
    # get the data from get_camera_calibration_primary_mono
    (
        K_rgb,
        D_rgb,
        fx_rgb,
        fy_rgb,
        cx_rgb,
        cy_rgb,
        K_left,
        D_left,
        fx_left,
        fy_left,
        cx_left,
        cy_left,
        K_right,
        D_right,
        fx_right,
        fy_right,
        cx_right,
        cy_right,
        rgb_fov,
        mono_fov,
        R1,
        R2,
        T1,
        T2,
        H_left,
        H_right,
        l2r_extrinsic,
        r2l_extrinsic,
        baseline,
        K_primary,
        D_primary,
        fx_primary,
        fy_primary,
        cx_primary,
        cy_primary,
        R_primary,
        T_primary,
        primary_extrinsic,
    ) = get_camera_calibration_primary_mono(
        rgb_size=rgb_size,
        mono_size=mono_size,
        is_primary_mono_left=is_primary_mono_left,
    )

    # construct the Q matrix
    Q_primary = create_q_matrix(
        fx_primary, fy_primary, cx_primary, cy_primary, baseline
    )
    Q_left = create_q_matrix(fx_left, fy_left, cx_left, cy_left, baseline)
    Q_right = create_q_matrix(fx_right, fy_right, cx_right, cy_right, baseline)

    # run cv2.getOptimalNewCameraMatrix for RGB cam
    P_rgb, valid_region_rgb = cv2.getOptimalNewCameraMatrix(
        K_rgb,
        D_rgb,
        rgb_size,
        1,
        rgb_size,
    )
    map_rgb_1, map_rgb_2 = cv2.initUndistortRectifyMap(
        K_rgb,
        D_rgb,
        None,
        P_rgb,
        rgb_size,
        cv2.CV_16SC2,
    )
    pinhole_rgb = o3d.camera.PinholeCameraIntrinsic(
        width=rgb_size[0],
        height=rgb_size[1],
        fx=fx_rgb,
        fy=fy_rgb,
        cx=cx_rgb,
        cy=cy_rgb,
    )

    # construct the ColorCalibrationData object
    color_calibration_data = ColorCalibrationData(
        size=rgb_size,
        K=K_rgb,
        D=D_rgb,
        fx=fx_rgb,
        fy=fy_rgb,
        cx=cx_rgb,
        cy=cy_rgb,
        fov=rgb_fov,
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
        K_left,
        D_left,
        K_right,
        D_right,
        mono_size,
        R_primary,
        T_primary,
    )
    map_left_1, map_left_2 = cv2.initUndistortRectifyMap(
        K_left,
        D_left,
        cv2_R1,
        P1,
        mono_size,
        cv2.CV_16SC2,
    )
    map_right_1, map_right_2 = cv2.initUndistortRectifyMap(
        K_right,
        D_right,
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

    # create open3d PinholeCameraIntrinsic objects for left, right, and primary mono cams
    pinhole_left = o3d.camera.PinholeCameraIntrinsic(
        width=mono_size[0],
        height=mono_size[1],
        fx=fx_left,
        fy=fy_left,
        cx=cx_left,
        cy=cy_left,
    )
    pinhole_right = o3d.camera.PinholeCameraIntrinsic(
        width=mono_size[0],
        height=mono_size[1],
        fx=fx_right,
        fy=fy_right,
        cx=cx_right,
        cy=cy_right,
    )
    pinhole_primary = pinhole_left if is_primary_mono_left else pinhole_right

    # construct the left, right, and primary MonoCalibrationData objects
    left_mono_calibration_data = MonoCalibrationData(
        size=mono_size,
        K=K_left,
        D=D_left,
        fx=fx_left,
        fy=fy_left,
        cx=cx_left,
        cy=cy_left,
        fov=mono_fov,
        R=R1,
        T=T1,
        H=H_left,
        valid_region=valid_region_left,
        map_1=map_left_1,
        map_2=map_left_2,
        pinhole=pinhole_left,
    )
    right_mono_calibration_data = MonoCalibrationData(
        size=mono_size,
        K=K_right,
        D=D_right,
        fx=fx_right,
        fy=fy_right,
        cx=cx_right,
        cy=cy_right,
        fov=mono_fov,
        R=R2,
        T=T2,
        H=H_right,
        valid_region=valid_region_right,
        map_1=map_right_1,
        map_2=map_right_2,
        pinhole=pinhole_right,
    )
    primary_mono_calibration_data = MonoCalibrationData(
        size=mono_size,
        K=K_primary,
        D=D_primary,
        fx=fx_primary,
        fy=fy_primary,
        cx=cx_primary,
        cy=cy_primary,
        fov=mono_fov,
        R=R_primary,
        T=T_primary,
        H=primary_extrinsic,
        valid_region=valid_region_primary,
        map_1=map_1_primary,
        map_2=map_2_primary,
        pinhole=pinhole_primary,
    )

    # construct the StereoCalibrationData object
    stereo_calibration_data = StereoCalibrationData(
        left=left_mono_calibration_data,
        right=right_mono_calibration_data,
        R1=R1,
        R2=R2,
        T1=T1,
        T2=T2,
        H_left=H_left,
        H_right=H_right,
        l2r_extrinsic=l2r_extrinsic,
        r2l_extrinsic=r2l_extrinsic,
        Q_left=Q_left,
        Q_right=Q_right,
        baseline=baseline,
        primary=primary_mono_calibration_data,
        Q_primary=Q_primary,
        cv2_Q=cv2_Q,
        cv2_R1=cv2_R1,
        cv2_R2=cv2_R2,
        P1=P1,
        P2=P2,
        valid_region_primary=valid_region_primary,
        pinhole_primary=pinhole_primary,
    )

    # construct the CalibrationData object
    calibration_data = CalibrationData(
        rgb=color_calibration_data,
        left=left_mono_calibration_data,
        right=right_mono_calibration_data,
        stereo=stereo_calibration_data,
        primary=primary_mono_calibration_data,
    )

    return calibration_data
