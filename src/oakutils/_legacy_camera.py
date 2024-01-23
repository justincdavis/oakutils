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
Module for interacting with the OAK-D enabling easy access to the RGB, depth, and IMU sensors.

Classes
-------
LegacyCamera
    Class for interfacing with the OAK-D camera with fixed pipeline.
"""
from __future__ import annotations

import atexit
import contextlib
from threading import Condition, Thread
from typing import TYPE_CHECKING

import cv2  # type: ignore[import]
import depthai as dai
import numpy as np

from .calibration import CalibrationData, get_camera_calibration
from .nodes import create_color_camera, create_imu, create_stereo_depth, create_xout
from .point_clouds import (
    PointCloudVisualizer,
    filter_point_cloud,
    get_point_cloud_from_rgb_depth_image,
)
from .tools.parsing import (
    get_color_sensor_info_from_str,
    get_median_filter_from_str,
    get_mono_sensor_info_from_str,
)

if TYPE_CHECKING:
    import open3d as o3d  # type: ignore[import]
    from typing_extensions import Self


# KNOWN BUGS:
# - Enabling the speckle filter crashes the camera
class LegacyCamera:
    """
    Class for interfacing with the OAK-D camera.

    Attributes
    ----------
    calibration : CalibrationData
        The calibration data for the camera.
    rgb : Optional[np.ndarray]
        The most recent RGB image from the camera.
    rectified_rgb : Optional[np.ndarray]
        The most recent rectified RGB image from the camera.
    disparity: Optional[np.ndarray]
        The most recent disparity image from the camera.
    depth: Optional[np.ndarray]
        The most recent depth image from the camera.
    left: Optional[np.ndarray]
        The most recent left mono image from the camera.
    right: Optional[np.ndarray]
        The most recent right mono image from the camera.
    rectified_left: Optional[np.ndarray]
        The most recent rectified left mono image from the camera.
    rectified_right: Optional[np.ndarray]
        The most recent rectified right mono image from the camera.
    im3d: Optional[np.ndarray]
        The most recent im3d image from the camera.
    point_cloud: Optional[o3d.geometry.PointCloud]
        The most recent point cloud from the camera.
    imu_pose: Optional[List[float]]
        The most recent IMU pose from the camera.
    imu_rotation: Optional[List[float]]
        The most recent IMU rotation from the camera.
    started: bool
        Whether or not the camera has been started.

    Methods
    -------
    start()
        Starts the camera.
    stop()
        Stops the camera.
    wait_for_data()
        Waits for the data packet to be ready.
    start_display()
        Starts the display.
    stop_display()
        Stops the display.
    compute_point_cloud(block=True)
        Computes the point cloud from the depth map.
    compute_im3d(block=True)
        Computes the 3D points from the disparity map.
    """

    def __init__(
        self: Self,
        rgb_size: str = "1080p",
        mono_size: str = "400p",
        rgb_fps: int = 30,
        mono_fps: int = 60,
        display_size: tuple[int, int] = (640, 400),
        median_filter: int | None = 7,
        stereo_confidence_threshold: int = 200,
        stereo_speckle_filter_range: int = 60,
        stereo_spatial_filter_radius: int = 2,
        stereo_spatial_filter_num_iterations: int = 1,
        stereo_threshold_filter_min_range: int = 200,
        stereo_threshold_filter_max_range: int = 20000,
        stereo_decimation_filter_factor: int = 1,
        imu_batch_report_threshold: int = 20,
        imu_max_batch_reports: int = 20,
        imu_accelerometer_refresh_rate: int = 400,
        imu_gyroscope_refresh_rate: int = 400,
        *,
        enable_rgb: bool | None = None,
        enable_mono: bool | None = None,
        primary_mono_left: bool | None = None,
        use_cv2_q_matrix: bool | None = None,
        compute_im3d_on_demand: bool | None = None,
        compute_point_cloud_on_demand: bool | None = None,
        display_rgb: bool | None = None,
        display_mono: bool | None = None,
        display_depth: bool | None = None,
        display_disparity: bool | None = None,
        display_rectified: bool | None = None,
        display_point_cloud: bool | None = None,
        extended_disparity: bool | None = None,
        subpixel: bool | None = None,
        lr_check: bool | None = None,
        stereo_speckle_filter_enable: bool | None = None,
        stereo_temporal_filter_enable: bool | None = None,
        stereo_spatial_filter_enable: bool | None = None,
        enable_imu: bool | None = None,
    ) -> None:
        """
        Use to create the camera object.

        Parameters
        ----------
        rgb_size : str, optional
            Size of the RGB image. Options are 1080p, 4K.
        enable_rgb : bool, optional
            Whether to enable the RGB camera.
        mono_size : str, optional
            Size of the monochrome image. Options are 720p, 480p, 400p.
        enable_mono : bool, optional
            Whether to enable the monochrome camera.
        rgb_fps : int, optional
            FPS for the RGB camera.
        mono_fps : int, optional
            FPS for the monochrome camera.
        primary_mono_left : bool, optional
            Whether the primary monochrome image is the left image or the right image.
        use_cv2_q_matrix : bool, optional
            Whether to use the cv2.Q matrix for disparity to depth conversion.
        compute_im3d_on_demand : bool, optional
            Whether to compute the IM3D on update.
        compute_point_cloud_on_demand : bool, optional
            Whether to compute the point cloud on update.
        display_size : tuple[int, int], optional
            Size of the display window.
        display_rgb : bool, optional
            Whether to display the RGB image.
        display_mono : bool, optional
            Whether to display the monochrome image.
        display_depth : bool, optional
            Whether to display the depth image.
        display_disparity : bool, optional
            Whether to display the disparity image.
        display_rectified : bool, optional
            Whether to display the rectified image.
        display_point_cloud : bool, optional
            Whether to display the point cloud.
        extended_disparity : bool, optional
            Whether to use extended disparity.
        subpixel : bool, optional
            Whether to use subpixel.
        lr_check : bool, optional
            Whether to use left-right check.
        median_filter : int or None, optional
            Whether to use median filter. If so, what size.
        stereo_confidence_threshold : int, optional
            Confidence threshold for stereo matching.
        stereo_speckle_filter_enable : bool, optional
            Whether to use speckle filter.
        stereo_speckle_filter_range : int, optional
            Speckle filter range.
        stereo_temporal_filter_enable : bool, optional
            Whether to use temporal filter.
        stereo_spatial_filter_enable : bool, optional
            Whether to use spatial filter.
        stereo_spatial_filter_radius : int, optional
            Spatial filter radius.
        stereo_spatial_filter_num_iterations : int, optional
            Spatial filter number of iterations.
        stereo_threshold_filter_min_range : int, optional
            Threshold filter minimum range.
        stereo_threshold_filter_max_range : int, optional
            Threshold filter maximum range.
        stereo_decimation_filter_factor : int, optional
            Decimation filter factor. Options are 1, 2.
        enable_imu : bool, optional
            Whether to enable the IMU.
        imu_batch_report_threshold : int, optional
            IMU batch report threshold.
        imu_max_batch_reports : int, optional
            IMU maximum report batches.
        imu_accelerometer_refresh_rate : int, optional
            IMU accelerometer refresh rate.
        imu_gyroscope_refresh_rate : int, optional
            IMU gyroscope refresh rate.
        """
        if enable_rgb is None:
            enable_rgb = True
        if enable_mono is None:
            enable_mono = True
        if primary_mono_left is None:
            primary_mono_left = True
        if use_cv2_q_matrix is None:
            use_cv2_q_matrix = True
        if compute_im3d_on_demand is None:
            compute_im3d_on_demand = True
        if compute_point_cloud_on_demand is None:
            compute_point_cloud_on_demand = True
        if display_rgb is None:
            display_rgb = False
        if display_mono is None:
            display_mono = False
        if display_depth is None:
            display_depth = False
        if display_disparity is None:
            display_disparity = False
        if display_rectified is None:
            display_rectified = False
        if display_point_cloud is None:
            display_point_cloud = False
        if extended_disparity is None:
            extended_disparity = True
        if subpixel is None:
            subpixel = False
        if lr_check is None:
            lr_check = True
        if stereo_speckle_filter_enable is None:
            stereo_speckle_filter_enable = False
        if stereo_temporal_filter_enable is None:
            stereo_temporal_filter_enable = True
        if stereo_spatial_filter_enable is None:
            stereo_spatial_filter_enable = True
        if enable_imu is None:
            enable_imu = False

        self._nodes: list[dai.Node] = []

        self._primary_mono_left = primary_mono_left
        self._use_cv2_q_matrix = use_cv2_q_matrix

        self._display_size = display_size
        self._display_rgb = display_rgb
        self._display_mono = display_mono
        self._display_depth = display_depth
        self._display_disparity = display_disparity
        self._display_rectified = display_rectified
        self._display_point_cloud = display_point_cloud

        self._stereo_confidence_threshold = stereo_confidence_threshold

        self._rgb_size = get_color_sensor_info_from_str(rgb_size)
        self._mono_size = get_mono_sensor_info_from_str(mono_size)

        dec_filter_divisor = 2
        if stereo_decimation_filter_factor == dec_filter_divisor:
            # need to divide the mono height by 2
            self._mono_size = (
                self._mono_size[0],
                self._mono_size[1] // dec_filter_divisor,
                self._mono_size[2],
            )

        self._median_filter = get_median_filter_from_str(median_filter)

        self._calibration: CalibrationData = get_camera_calibration(
            (self._rgb_size[0], self._rgb_size[1]),
            (self._mono_size[0], self._mono_size[1]),
            is_primary_mono_left=self._primary_mono_left,
        )
        self._Q = (
            self._calibration.stereo.Q_cv2
            if self._use_cv2_q_matrix
            else self._calibration.stereo.Q_primary
        )

        # pipeline
        self._pipeline: dai.Pipeline = dai.Pipeline()
        # storage for the nodes
        self._streams: list[str] = []
        # stop condition
        self._stopped: bool = False
        # thread for the camera
        self._cam_thread = Thread(target=self._target)

        self._rgb_frame: np.ndarray | None = None
        self._rectified_rgb_frame: np.ndarray | None = None
        self._disparity: np.ndarray | None = None
        self._depth: np.ndarray | None = None
        self._left_frame: np.ndarray | None = None
        self._right_frame: np.ndarray | None = None
        self._left_rect_frame: np.ndarray | None = None
        self._right_rect_frame: np.ndarray | None = None
        self._primary_rect_frame: np.ndarray | None = None

        self._im3d: np.ndarray | None = None
        self._compute_im3d_on_demand = compute_im3d_on_demand
        self._im3d_current = False

        self._point_cloud: o3d.geometry.PointCloud | None = None
        self._compute_point_cloud_on_demand = compute_point_cloud_on_demand
        if self._display_point_cloud:
            self._point_cloud_vis = PointCloudVisualizer()

        # imu information
        self._imu_packet: np.ndarray | None = None
        self._imu_batch_report_threshold: int = imu_batch_report_threshold
        self._imu_max_batch_reports: int = imu_max_batch_reports
        self._imu_accelerometer_refresh_rate: float = imu_accelerometer_refresh_rate
        self._imu_gyroscope_refresh_rate: float = imu_gyroscope_refresh_rate
        self._imu_pose: list[float] = [0, 0, 0]
        self._imu_rotation: list[float] = [0, 0, 0]

        # packet for compute_3d
        self._3d_packet: tuple[
            np.ndarray | None,
            np.ndarray | None,
            np.ndarray | None,
        ] = (None, None, None)

        # display information
        self._display_thread = Thread(target=self._display)
        self._display_stopped = False

        # Condition for data
        self._data_condition = Condition()

        # creaate the nodes on the pipeline
        if enable_rgb:
            cam = create_color_camera(
                pipeline=self._pipeline,
                resolution=self._rgb_size[2],
                fps=rgb_fps,
            )
            create_xout(self._pipeline, cam.video, "rgb")
            self._streams.extend(["rgb"])
            self._nodes.extend([cam])
        if enable_mono:
            align_socket = (
                dai.CameraBoardSocket.LEFT
                if self._primary_mono_left
                else dai.CameraBoardSocket.RIGHT
            )
            (
                stereo,
                left,
                right,
            ) = create_stereo_depth(
                pipeline=self._pipeline,
                resolution=self._mono_size[2],
                fps=mono_fps,
                align_socket=align_socket,
                confidence_threshold=stereo_confidence_threshold,
                median_filter=self._median_filter,
                lr_check=lr_check,
                extended_disparity=extended_disparity,
                subpixel=subpixel,
                decimation_factor=stereo_decimation_filter_factor,
                enable_spatial_filter=stereo_spatial_filter_enable,
                enable_speckle_filter=stereo_speckle_filter_enable,
                enable_temporal_filter=stereo_temporal_filter_enable,
                speckle_range=stereo_speckle_filter_range,
                spatial_radius=stereo_spatial_filter_radius,
                spatial_iterations=stereo_spatial_filter_num_iterations,
                threshold_min_range=stereo_threshold_filter_min_range,
                threshold_max_range=stereo_threshold_filter_max_range,
            )
            create_xout(self._pipeline, stereo.syncedLeft, "left")
            create_xout(self._pipeline, stereo.syncedRight, "right")
            create_xout(self._pipeline, stereo.depth, "depth")
            create_xout(self._pipeline, stereo.disparity, "disparity")
            create_xout(self._pipeline, stereo.rectifiedLeft, "rectified_left")
            create_xout(self._pipeline, stereo.rectifiedRight, "rectified_right")

            self._streams.extend(
                [
                    "left",
                    "right",
                    "depth",
                    "disparity",
                    "rectified_left",
                    "rectified_right",
                ],
            )
            self._nodes.extend([stereo, left, right])
        if enable_imu:
            imu = create_imu(
                pipeline=self._pipeline,
                accelerometer_rate=self._imu_accelerometer_refresh_rate,
                gyroscope_rate=self._imu_gyroscope_refresh_rate,
                batch_report_threshold=self._imu_batch_report_threshold,
                max_batch_reports=self._imu_max_batch_reports,
                enable_accelerometer_raw=True,
                enable_gyroscope_raw=True,
            )
            create_xout(self._pipeline, imu.out, "imu")

            self._streams.extend(["imu"])
            self._nodes.extend([imu])

        # set atexit methods
        atexit.register(self.stop)

    @property
    def calibration(self: Self) -> CalibrationData:
        """
        Gets the calibration data.

        Returns
        -------
        np.ndarray
            The calibration data.
        """
        return self._calibration

    @property
    def rgb(self: Self) -> np.ndarray | None:
        """
        Get the rectified RGB color frame.

        Returns
        -------
        Optional[np.ndarray]
            The rectified RGB color frame, or None if the frame is not available.
        """
        return self._rgb_frame

    @property
    def rectified_rgb(self: Self) -> np.ndarray | None:
        """
        Get the rectified RGB color frame.

        Returns
        -------
        Optional[np.ndarray]
            The rectified RGB color frame, or None if the frame is not available.
        """
        return self._rectified_rgb_frame

    @property
    def disparity(self: Self) -> np.ndarray | None:
        """
        Get the disparity frame.

        Returns
        -------
        Optional[np.ndarray]
            The disparity frame, or None if the frame is not available.
        """
        return self._disparity

    @property
    def depth(self: Self) -> np.ndarray | None:
        """
        Get the depth frame.

        Returns
        -------
        Optional[np.ndarray]
            The depth frame, or None if the frame is not available.
        """
        return self._depth

    @property
    def left(self: Self) -> np.ndarray | None:
        """
        Get the left frame.

        Returns
        -------
        Optional[np.ndarray]
            The left frame, or None if the frame is not available.
        """
        return self._left_frame

    @property
    def right(self: Self) -> np.ndarray | None:
        """
        Get the right frame.

        Returns
        -------
        Optional[np.ndarray]
            The right frame, or None if the frame is not available.
        """
        return self._right_frame

    @property
    def rectified_left(self: Self) -> np.ndarray | None:
        """
        Gets the rectified left frame.

        Returns
        -------
        Optional[np.ndarray]
            The rectified left frame, or None if the frame is not available.
        """
        return self._left_rect_frame

    @property
    def rectified_right(self: Self) -> np.ndarray | None:
        """
        Gets the rectified right frame.

        Returns
        -------
        Optional[np.ndarray]
            The rectified right frame, or None if the frame is not available.
        """
        return self._right_rect_frame

    @property
    def im3d(self: Self) -> np.ndarray | None:
        """
        Gets the 3D image.

        Returns
        -------
        Optional[np.ndarray]
            The 3D image, or None if it is not available.
        """
        return self._im3d

    @property
    def point_cloud(self: Self) -> o3d.geometry.PointCloud | None:
        """
        Gets the point cloud.

        Returns
        -------
        Optional[o3d.geometry.PointCloud]
            The point cloud, or None if it is not available.
        """
        return self._point_cloud

    @property
    def imu_pose(self: Self) -> list[float]:
        """
        Gets the IMU pose in meters.

        Returns
        -------
        List[float]
            The IMU pose as a list of floats.
        """
        return self._imu_pose

    @property
    def imu_rotation(self: Self) -> list[float]:
        """
        Gets the IMU rotation in radians.

        Returns
        -------
        List[float]
            The IMU rotation as a list of floats.
        """
        return self._imu_rotation

    @property
    def started(self: Self) -> bool:
        """
        Returns True if the camera is started.

        Returns
        -------
        bool
            True if the camera is started, False otherwise.
        """
        return self._cam_thread.is_alive()

    def start(self: Self, *, block: bool | None = None) -> None:
        """
        Use to start the camera.

        Parameters
        ----------
        block : bool, optional
            If True, blocks until the first set of data arrives. Defaults to False.
        """
        if block is None:
            block = True
        self._cam_thread.start()
        if block:
            self.wait_for_data()

    def stop(self: Self) -> None:
        """Use to stop the camera."""
        self._stopped = True
        with contextlib.suppress(RuntimeError):
            self._cam_thread.join()

        # stop the displays
        self._display_stopped = True
        with contextlib.suppress(RuntimeError):
            self._display_thread.join()

        # close displays
        cv2.destroyAllWindows()

    def wait_for_data(self: Self) -> None:
        """Blocks until a full set of data has arrived from the camera."""
        with self._data_condition:
            self._data_condition.wait()

    def _display(self: Self) -> None:
        with self._data_condition:
            self._data_condition.wait()
        while not self._display_stopped:
            if self._rgb_frame is not None and self._display_rgb:
                cv2.imshow("rgb", cv2.resize(self._rgb_frame, self._display_size))
            if self._disparity is not None and self._display_disparity:
                frame: np.ndarray = (
                    self._disparity * (255 / self._stereo_confidence_threshold)
                ).astype(np.uint8)
                frame = cv2.resize(frame, self._display_size)
                cv2.imshow("disparity-gray", frame)
                frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
                cv2.imshow("disparity-color", frame)
            if self._depth is not None and self._display_depth:
                cv2.imshow("depth", cv2.resize(self._depth, self._display_size))
            if self._left_frame is not None and self._display_mono:
                cv2.imshow("left", cv2.resize(self._left_frame, self._display_size))
            if self._right_frame is not None and self._display_mono:
                cv2.imshow("right", cv2.resize(self._right_frame, self._display_size))
            if self._left_rect_frame is not None and self._display_rectified:
                cv2.imshow(
                    "rectified left",
                    cv2.resize(self._left_rect_frame, self._display_size),
                )
            if self._right_rect_frame is not None and self._display_rectified:
                cv2.imshow(
                    "rectified right",
                    cv2.resize(self._right_rect_frame, self._display_size),
                )
            if self._point_cloud is not None and self._display_point_cloud:
                self._point_cloud_vis.update(self._point_cloud)
            cv2.waitKey(50)
        if self._display_point_cloud:
            self._point_cloud_vis.stop()

    def start_display(self: Self) -> None:
        """Use to start the display thread."""
        self._display_thread.start()

    def stop_display(self: Self) -> None:
        """Use to stop the display thread."""
        self._display_stopped = True
        self._display_thread.join()

    def _update_point_cloud(self: Self) -> None:
        if self._rgb_frame is None or self._depth is None:
            err_msg = "RGB frame or depth map is not available."
            raise RuntimeError(err_msg)
        if (
            self._calibration.primary is None
            or self._calibration.primary.pinhole is None
        ):
            err_msg = "Primary pinhole calibration is not available."
            raise RuntimeError(err_msg)

        pcd = get_point_cloud_from_rgb_depth_image(
            self._rgb_frame,
            self._depth,
            self._calibration.primary.pinhole,
        )

        pcd = filter_point_cloud(
            pcd,
            voxel_size=None,
            nb_neighbors=30,
            std_ratio=0.1,
            downsample_first=True,
        )

        if self._point_cloud is None:
            self._point_cloud = pcd
        else:
            self._point_cloud.points = pcd.points
            self._point_cloud.colors = pcd.colors

    def _update_im3d(self: Self) -> None:
        self._im3d = cv2.reprojectImageTo3D(self._disparity, self._Q)  # type: ignore[arg-type]

    def _target(self: Self) -> None:
        with dai.Device(self._pipeline) as device:
            queues = {}
            for stream in self._streams:
                queues[stream] = device.getOutputQueue(  # type: ignore[attr-defined]
                    name=stream,
                    maxSize=1,
                    blocking=False,
                )

            base_accel_timestamp = None
            base_gyro_timestamp = None
            while not self._stopped:
                for name, queue in queues.items():
                    if queue is not None:
                        data = queue.get()
                        if name == "rgb":
                            self._rgb_frame = data.getCvFrame()
                            self._rectified_rgb_frame = cv2.remap(
                                self._rgb_frame,  # type: ignore[arg-type]
                                self._calibration.rgb.map_1,  # type: ignore[arg-type]
                                self._calibration.rgb.map_2,  # type: ignore[arg-type]
                                cv2.INTER_LINEAR,
                            )
                        elif name == "left":
                            self._left_frame = data.getCvFrame()
                        elif name == "right":
                            self._right_frame = data.getCvFrame()
                        elif name == "depth":
                            self._depth = data.getCvFrame()
                        elif name == "disparity":
                            self._disparity = data.getCvFrame()
                        elif name == "rectified_left":
                            self._left_rect_frame = data.getCvFrame()
                        elif name == "rectified_right":
                            self._right_rect_frame = data.getCvFrame()
                        elif name == "imu":
                            packets = data.packets
                            for imu_packet in packets:
                                acc_values = imu_packet.acceleroMeter
                                gyro_values = imu_packet.gyroscope

                                acclero_ts_device = acc_values.getTimestampDevice()
                                gyro_ts_device = gyro_values.getTimestampDevice()

                                if base_accel_timestamp is None:
                                    base_accel_timestamp = acclero_ts_device
                                if base_gyro_timestamp is None:
                                    base_gyro_timestamp = gyro_ts_device

                                accelero_ts = (
                                    acclero_ts_device - base_accel_timestamp
                                ).total_seconds()
                                gyro_ts = (
                                    gyro_ts_device - base_gyro_timestamp
                                ).total_seconds()

                                ax, ay, az = (
                                    acc_values.x,
                                    acc_values.y,
                                    acc_values.z,
                                )
                                gx, gy, gz = gyro_values.x, gyro_values.y, gyro_values.z

                                # double integrate each ax, ay, az
                                self._imu_pose[0] += ax * (accelero_ts**2)
                                self._imu_pose[1] += ay * (accelero_ts**2)
                                self._imu_pose[2] += az * (accelero_ts**2)

                                # integrate each gx, gy, gz
                                self._imu_rotation[0] += gx * gyro_ts
                                self._imu_rotation[1] += gy * gyro_ts
                                self._imu_rotation[2] += gz * gyro_ts

                                base_accel_timestamp = acclero_ts_device
                                base_gyro_timestamp = gyro_ts_device

                # handle primary mono camera
                self._primary_rect_frame = (
                    self._left_rect_frame
                    if self._primary_mono_left
                    else self._right_rect_frame
                )
                # handle 3d images and odometry packets
                if not self._compute_im3d_on_demand:
                    self._update_im3d()
                self._3d_packet = (
                    self._im3d,
                    self._disparity,
                    self._primary_rect_frame,
                )
                # handle the point cloud
                if not self._compute_point_cloud_on_demand:
                    self._update_point_cloud()

                with self._data_condition:
                    self._data_condition.notify_all()

    def _crop_to_valid_primary_region(self: Self, img: np.ndarray) -> np.ndarray:
        if self._calibration.primary is None:
            err_msg = "Primary calibration is not available."
            raise RuntimeError(err_msg)
        if self._calibration.primary.valid_region is None:
            err_msg = "Primary valid region is not available."
            raise RuntimeError(err_msg)
        return img[
            self._calibration.primary.valid_region[
                1
            ] : self._calibration.primary.valid_region[3],
            self._calibration.primary.valid_region[
                0
            ] : self._calibration.primary.valid_region[2],
        ]

    def compute_point_cloud(
        self: Self,
        *,
        block: bool | None = None,
    ) -> o3d.geometry.PointCloud | None:
        """
        Compute a point cloud from the depth map.

        Parameters
        ----------
        block : bool, optional
            If True, blocks until the next data packet is received. Defaults to True.

        Returns
        -------
        Optional[o3d.geometry.PointCloud]
            The computed point cloud, or None if no data is available.
        """
        if block is None:
            block = True
        if block:
            with self._data_condition:
                self._data_condition.wait()
        if self._rgb_frame is None and self._depth is None:
            return None
        if self._compute_point_cloud_on_demand:
            self._update_point_cloud()
        return self._point_cloud

    def compute_im3d(
        self: Self,
        *,
        block: bool | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """
        Compute 3D points from the disparity map.

        Parameters
        ----------
        block : bool, optional
            If True, blocks until the next data packet is received. Defaults to True.

        Returns
        -------
        Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]
            A tuple containing the depth map, disparity map, and left frame
            (if available).
        """
        if block is None:
            block = True
        if block:
            with self._data_condition:
                self._data_condition.wait()
        im3d, disparity, rect = self._3d_packet
        if im3d is None and disparity is None and rect is None:
            return None, None, None
        if self._compute_im3d_on_demand:
            self._update_im3d()
            im3d = self._im3d
        if im3d is None or disparity is None or rect is None:
            return None, None, None
        return (
            self._crop_to_valid_primary_region(im3d),
            self._crop_to_valid_primary_region(disparity),
            self._crop_to_valid_primary_region(rect),
        )
