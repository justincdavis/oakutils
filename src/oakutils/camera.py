from threading import Thread, Condition
from typing import List, Tuple, Optional, Union, Dict, Callable
import atexit

import depthai as dai
import depthai_sdk as sdk
import numpy as np
import cv2
import open3d as o3d

from .calibration import CalibrationData, get_camera_calibration
from .point_clouds import (
    PointCloudVisualizer,
    get_point_cloud_from_rgb_depth_image,
    filter_point_cloud,
)
from .nodes import (
    create_color_camera,
    create_stereo_depth,
    create_imu,
)
from .tools.parsing import (
    get_color_sensor_info_from_str,
    get_mono_sensor_info_from_str,
    get_median_filter_from_str,
)
from .tools.display import DisplayManager, get_smaller_size


class OakCamera:
    def __init__(
        self,
        # standard sdk.OakCamera args
        device: Optional[str] = None,
        usb_speed: Optional[Union[str, dai.UsbSpeed]] = None,
        replay: Optional[str] = None,
        rotation: Optional[int] = None,
        config: Optional[dai.Device.Config] = None,
        # custom args
        primary_mono_left: bool = True,
        color_size: Tuple[int, int] = (1920, 1080),
        mono_size: Tuple[int, int] = (640, 400),
    ):
        # store custom args
        self._color_size: Tuple[int, int] = color_size
        self._mono_size: Tuple[int, int] = mono_size
        self._primary_mono_left: bool = primary_mono_left

        # handle attributes
        self._calibration: CalibrationData = get_camera_calibration(
            self._color_size, self._mono_size, self._primary_mono_left
        )
        self._callbacks: Dict[str, Callable] = {}
        self._pipeline: Optional[dai.Pipeline] = None
        self._is_built: bool = False

        # store oak camera
        self._oak: sdk.OakCamera = sdk.OakCamera(
            device=device,
            usb_speed=usb_speed,
            replay=replay,
            rotation=rotation,
            config=config,
            args=True,
        )

        # handle custom displays directly for API stuff without visualize
        self._displays: DisplayManager = DisplayManager(
            display_size=get_smaller_size(self._color_size, self._mono_size)
        )

    @property
    def oak(self) -> sdk.OakCamera:
        """
        Returns the OakCamera object.
        """
        return self._oak

    @property
    def pipeline(self) -> dai.Pipeline:
        """
        Returns the pipeline. If the pipeline has not been built yet, it will be built.
        """
        if self._pipeline is None:
            self._pipeline: dai.Pipeline = self._oak.build()
            self._is_built = True
        return self._pipeline

    @property
    def calibration(self) -> CalibrationData:
        """
        Returns the calibration data.
        """
        return self._calibration

    @property
    def displays(self) -> DisplayManager:
        """
        Returns the display manager.
        """
        return self._displays
