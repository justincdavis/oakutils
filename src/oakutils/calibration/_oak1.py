# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import cv2  # type: ignore[import]
import depthai as dai
import numpy as np

try:
    import open3d as o3d  # type: ignore[import]

    PinholeCameraIntrinsic = o3d.camera.PinholeCameraIntrinsic
except ImportError:
    PinholeCameraIntrinsic = None

from ._classes import ColorCalibrationData


def get_oak1_calibration_basic(
    device: dai.DeviceBase | None = None,
    rgb_size: tuple[int, int] = (1920, 1080),
) -> ColorCalibrationData:
    """
    Use to get camera calibration data from OAK-1 device.

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

        rgb_fov = calib_data.getFov(dai.CameraBoardSocket.RGB)
        rgb_fov_rad = np.deg2rad(rgb_fov)

        return ColorCalibrationData(
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


def get_oak1_calibration(
    rgb_size: tuple[int, int],
    device: dai.DeviceBase | None = None,
) -> ColorCalibrationData:
    """
    Use to create the full ColorCalibrationData object.

    This includes the calibration data for the RGB camera.

    Note
    ----
        Requires available OAK device.
        If device is not provided, dai.Device() will be used.

    Parameters
    ----------
    rgb_size : Tuple[int, int]
        RGB camera resolution.
    device : Optional[dai.Device], optional
        DepthAI device object.

    Returns
    -------
    ColorCalibrationData
        Object containing all the calibration data.

    """
    # get the data from get_oak1_calibration_basic
    data: ColorCalibrationData = get_oak1_calibration_basic(
        device=device,
        rgb_size=rgb_size,
    )

    # run cv2.getOptimalNewCameraMatrix for RGB cam
    p_rgb, valid_region_rgb = cv2.getOptimalNewCameraMatrix(
        data.K,
        data.D,
        rgb_size,
        1,
        rgb_size,
    )
    map_rgb_1, map_rgb_2 = cv2.initUndistortRectifyMap(
        data.K,
        data.D,
        None,  # pyright: ignore[reportArgumentType]
        p_rgb,
        rgb_size,
        cv2.CV_16SC2,  # type: ignore[attr-defined]
    )  # type: ignore[call-overload]
    pinhole_rgb = None
    if PinholeCameraIntrinsic is not None:
        pinhole_rgb = PinholeCameraIntrinsic(
            width=rgb_size[0],
            height=rgb_size[1],
            fx=data.fx,
            fy=data.fy,
            cx=data.cx,
            cy=data.cy,
        )

    # add the pinhole data and maps to the data.rgb object
    return ColorCalibrationData(
        size=data.size,
        K=data.K,
        D=data.D,
        fx=data.fx,
        fy=data.fy,
        cx=data.cx,
        cy=data.cy,
        fov=data.fov,
        fov_rad=data.fov_rad,
        P=p_rgb,
        valid_region=valid_region_rgb,  # type: ignore[arg-type]
        map_1=map_rgb_1,
        map_2=map_rgb_2,
        pinhole=pinhole_rgb,
    )
