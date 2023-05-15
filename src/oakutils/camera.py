from threading import Thread, Condition
from typing import List, Tuple, Optional
import atexit

import depthai as dai
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


# KNOWN BUGS:
# - Enabling the speckle filter crashes the camera
class Camera:
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
        self,
        rgb_size: str = "1080p",
        enable_rgb: bool = True,
        mono_size: str = "400p",
        enable_mono: bool = True,
        rgb_fps: int = 30,
        mono_fps: int = 60,
        primary_mono_left: bool = True,
        use_cv2_Q_matrix: bool = True,
        compute_im3d_on_demand: bool = True,
        compute_point_cloud_on_demand: bool = True,
        display_size: Tuple[int, int] = (640, 400),
        display_rgb: bool = False,
        display_mono: bool = False,
        display_depth: bool = False,
        display_disparity: bool = True,
        display_rectified: bool = False,
        display_point_cloud: bool = False,
        extended_disparity: bool = True,
        subpixel: bool = False,
        lr_check: bool = True,
        median_filter: Optional[int] = 7,
        stereo_confidence_threshold: int = 200,
        stereo_speckle_filter_enable: bool = False,
        stereo_speckle_filter_range: int = 60,
        stereo_temporal_filter_enable: bool = True,
        stereo_spatial_filter_enable: bool = True,
        stereo_spatial_filter_radius: int = 2,
        stereo_spatial_filter_num_iterations: int = 1,
        stereo_threshold_filter_min_range: int = 200,
        stereo_threshold_filter_max_range: int = 20000,
        stereo_decimation_filter_factor: int = 1,
        enable_imu: bool = False,
        imu_batch_report_threshold: int = 20,
        imu_max_batch_reports: int = 20,
        imu_accelerometer_refresh_rate: int = 400,
        imu_gyroscope_refresh_rate: int = 400,
    ):
        """
        Initializes the camera object.

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
        use_cv2_Q_matrix : bool, optional
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
        self._primary_mono_left = primary_mono_left
        self._use_cv2_Q_matrix = use_cv2_Q_matrix

        self._display_size = display_size
        self._display_rgb = display_rgb
        self._display_mono = display_mono
        self._display_depth = display_depth
        self._display_disparity = display_disparity
        self._display_rectified = display_rectified
        self._display_point_cloud = display_point_cloud

        self._stereo_confidence_threshold = stereo_confidence_threshold

        if rgb_size not in ["4k", "1080p"]:
            raise ValueError('rgb_size must be one of "1080p" or "4k"')
        else:
            if rgb_size == "4k":
                self._rgb_size = (
                    3840,
                    2160,
                    dai.ColorCameraProperties.SensorResolution.THE_4_K,
                )
            elif rgb_size == "1080p":
                self._rgb_size = (
                    1920,
                    1080,
                    dai.ColorCameraProperties.SensorResolution.THE_1080_P,
                )

        if mono_size not in ["720p", "480p", "400p"]:
            raise ValueError('mono_size must be one of "720p", "480p", or "400p"')
        else:
            if mono_size == "720p":
                self._mono_size = (
                    1280,
                    720,
                    dai.MonoCameraProperties.SensorResolution.THE_720_P,
                )
            elif mono_size == "480p":
                self._mono_size = (
                    640,
                    480,
                    dai.MonoCameraProperties.SensorResolution.THE_480_P,
                )
            elif mono_size == "400p":
                self._mono_size = (
                    640,
                    400,
                    dai.MonoCameraProperties.SensorResolution.THE_400_P,
                )

        if stereo_decimation_filter_factor == 2:
            # need to divide the mono height by 2
            self._mono_size = (
                self._mono_size[0],
                self._mono_size[1] // 2,
                self._mono_size[2],
            )

        if median_filter not in [0, 3, 5, 7] and median_filter is not None:
            raise ValueError("Unsupported median filter size, use 0, 3, 5, 7, or None")
        else:
            self._median_filter = median_filter
            if self._median_filter == 3:
                self._median_filter = dai.StereoDepthProperties.MedianFilter.KERNEL_3x3
            elif self._median_filter == 5:
                self._median_filter = dai.StereoDepthProperties.MedianFilter.KERNEL_5x5
            elif self._median_filter == 7:
                self._median_filter = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7
            else:
                self._median_filter = dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF

        self._calibration: CalibrationData = get_camera_calibration(
            (self._rgb_size[0], self._rgb_size[1]),
            (self._mono_size[0], self._mono_size[1]),
            self._primary_mono_left,
        )
        self._Q = (
            self._calibration.stereo.cv2_Q
            if self._use_cv2_Q_matrix
            else self._calibration.stereo.Q_primary
        )

        # pipeline
        self._pipeline: dai.Pipeline = dai.Pipeline()
        # storage for the nodes
        self._streams: List[str] = []
        # stop condition
        self._stopped: bool = False
        # thread for the camera
        self._cam_thread = Thread(target=self._target)

        self._rgb_frame: Optional[np.ndarray] = None
        self._rectified_rgb_frame: Optional[np.ndarray] = None
        self._disparity: Optional[np.ndarray] = None
        self._depth: Optional[np.ndarray] = None
        self._left_frame: Optional[np.ndarray] = None
        self._right_frame: Optional[np.ndarray] = None
        self._left_rect_frame: Optional[np.ndarray] = None
        self._right_rect_frame: Optional[np.ndarray] = None
        self._primary_rect_frame: Optional[np.ndarray] = None

        self._im3d: Optional[np.ndarray] = None
        self._compute_im3d_on_demand = compute_im3d_on_demand
        self._im3d_current = False

        self._point_cloud: Optional[o3d.geometry.PointCloud] = None
        self._compute_point_cloud_on_demand = compute_point_cloud_on_demand
        self._point_cloud_vis = PointCloudVisualizer()

        # imu information
        self._imu_packet: Optional[np.ndarray] = None
        self._imu_batch_report_threshold: int = imu_batch_report_threshold
        self._imu_max_batch_reports: int = imu_max_batch_reports
        self._imu_accelerometer_refresh_rate: float = imu_accelerometer_refresh_rate
        self._imu_gyroscope_refresh_rate: float = imu_gyroscope_refresh_rate
        self._imu_pose: List[float] = [0, 0, 0]
        self._imu_rotation: List[float] = [0, 0, 0]

        # packet for compute_3d
        self._3d_packet: Tuple[
            Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]
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
            xout_rgb = self._pipeline.create(dai.node.XLinkOut)
            xout_rgb.setStreamName("rgb")
            cam.video.link(xout_rgb.input)

            self._streams.extend(["rgb"])
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
                xout_left,
                xout_right,
                xout_depth,
                xout_disparity,
                xout_rect_left,
                xout_rect_right,
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

            self._streams.extend(
                [
                    "left",
                    "right",
                    "depth",
                    "disparity",
                    "rectified_left",
                    "rectified_right",
                ]
            )
        if enable_imu:
            imu, xout_imu = create_imu(
                pipeline=self._pipeline,
                accel_range=self._imu_accelerometer_refresh_rate,
                gyroscope_rate=self._imu_gyroscope_refresh_rate,
                batch_report_threshold=self._imu_batch_report_threshold,
                max_batch_reports=self._imu_max_batch_reports,
                enable_accelerometer_raw=True,
                enable_gyroscope_raw=True,
            )

            self._streams.extend(["imu"])

        # set atexit methods
        atexit.register(self.stop)

    @property
    def calibration(self) -> CalibrationData:
        """
        Gets the calibration data.

        Returns
        -------
        np.ndarray
            The calibration data.
        """
        return self._calibration

    @property
    def rgb(self) -> Optional[np.ndarray]:
        """
        Get the rectified RGB color frame.

        Returns
        -------
        Optional[np.ndarray]
            The rectified RGB color frame, or None if the frame is not available.
        """
        return self._rgb_frame

    @property
    def rectified_rgb(self) -> Optional[np.ndarray]:
        """
        Get the rectified RGB color frame.

        Returns
        -------
        Optional[np.ndarray]
            The rectified RGB color frame, or None if the frame is not available.
        """
        return self._rectified_rgb_frame

    @property
    def disparity(self) -> Optional[np.ndarray]:
        """
        Get the disparity frame.

        Returns
        -------
        Optional[np.ndarray]
            The disparity frame, or None if the frame is not available.
        """
        return self._disparity

    @property
    def depth(self) -> Optional[np.ndarray]:
        """
        Get the depth frame.

        Returns
        -------
        Optional[np.ndarray]
            The depth frame, or None if the frame is not available.
        """
        return self._depth

    @property
    def left(self) -> Optional[np.ndarray]:
        """
        Get the left frame.

        Returns
        -------
        Optional[np.ndarray]
            The left frame, or None if the frame is not available.
        """
        return self._left_frame

    @property
    def right(self) -> Optional[np.ndarray]:
        """
        Get the right frame.

        Returns
        -------
        Optional[np.ndarray]
            The right frame, or None if the frame is not available.
        """
        return self._right_frame

    @property
    def rectified_left(self) -> Optional[np.ndarray]:
        """
        Gets the rectified left frame.

        Returns
        -------
        Optional[np.ndarray]
            The rectified left frame, or None if the frame is not available.
        """
        return self._left_rect_frame

    @property
    def rectified_right(self) -> Optional[np.ndarray]:
        """
        Gets the rectified right frame.

        Returns
        -------
        Optional[np.ndarray]
            The rectified right frame, or None if the frame is not available.
        """
        return self._right_rect_frame

    @property
    def im3d(self) -> Optional[np.ndarray]:
        """
        Gets the 3D image.

        Returns
        -------
        Optional[np.ndarray]
            The 3D image, or None if it is not available.
        """
        return self._im3d

    @property
    def point_cloud(self) -> Optional[o3d.geometry.PointCloud]:
        """
        Gets the point cloud.

        Returns
        -------
        Optional[o3d.geometry.PointCloud]
            The point cloud, or None if it is not available.
        """
        return self._point_cloud

    @property
    def imu_pose(self) -> List[float]:
        """
        Gets the IMU pose in meters.

        Returns
        -------
        List[float]
            The IMU pose as a list of floats.
        """
        return self._imu_pose

    @property
    def imu_rotation(self) -> List[float]:
        """
        Gets the IMU rotation in radians.

        Returns
        -------
        List[float]
            The IMU rotation as a list of floats.
        """
        return self._imu_rotation

    @property
    def started(self) -> bool:
        """
        Returns True if the camera is started.

        Returns
        -------
        bool
            True if the camera is started, False otherwise.
        """
        return self._cam_thread.is_alive()

    def start(self, block=True) -> None:
        """
        Starts the camera.

        Parameters
        ----------
        block : bool, optional
            If True, blocks until the first set of data arrives. Defaults to False.
        """
        self._cam_thread.start()
        if block:
            self.wait_for_data()

    def stop(self) -> None:
        """
        Stops the camera
        """
        self._stopped = True
        try:
            self._cam_thread.join()
        except RuntimeError:
            pass

        # stop the displays
        self._display_stopped = True
        try:
            self._display_thread.join()
        except RuntimeError:
            pass

        # close displays
        cv2.destroyAllWindows()

    def wait_for_data(self) -> None:
        """
        Blocks until a full set of data has arrived from the camera
        """
        with self._data_condition:
            self._data_condition.wait()

    def _display(self) -> None:
        with self._data_condition:
            self._data_condition.wait()
        while not self._display_stopped:
            if self._rgb_frame is not None and self._display_rgb:
                cv2.imshow("rgb", cv2.resize(self._rgb_frame, self._display_size))
            if self._disparity is not None and self._display_disparity:
                frame = (
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
        self._point_cloud_vis.stop()

    def start_display(self) -> None:
        """
        Starts the display thread
        """
        self._display_thread.start()

    def stop_display(self) -> None:
        """
        Stops the display thread
        """
        self._display_stopped = True
        self._display_thread.join()

    def _update_point_cloud(self) -> None:
        pcd = get_point_cloud_from_rgb_depth_image(
            self._rgb_frame, self._depth, self._calibration.primary.pinhole
        )

        pcd = filter_point_cloud(
            pcd, voxel_size=None, nb_neighbors=30, std_ratio=0.1, downsample_first=True
        )

        if self._point_cloud is None:
            self._point_cloud = pcd
        else:
            self._point_cloud.points = pcd.points
            self._point_cloud.colors = pcd.colors

    def _update_im3d(self) -> None:
        self._im3d = cv2.reprojectImageTo3D(self._disparity, self._Q)

    def _target(self) -> None:
        with dai.Device(self._pipeline) as device:
            queues = {}
            for stream in self._streams:
                queues[stream] = device.getOutputQueue(
                    name=stream, maxSize=1, blocking=False
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
                                self._rgb_frame,
                                self._calibration.rgb.map_1,
                                self._calibration.rgb.map_2,
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
                            for imuPacket in packets:
                                acceleroValues = imuPacket.acceleroMeter
                                gyroValues = imuPacket.gyroscope

                                acclero_ts_device = acceleroValues.getTimestampDevice()
                                gyro_ts_device = gyroValues.getTimestampDevice()

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
                                    acceleroValues.x,
                                    acceleroValues.y,
                                    acceleroValues.z,
                                )
                                gx, gy, gz = gyroValues.x, gyroValues.y, gyroValues.z

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

    def _crop_to_valid_primary_region(self, img: np.ndarray) -> np.ndarray:
        return img[
            self._calibration.primary.valid_region[
                1
            ] : self._calibration.primary.valid_region[3],
            self._calibration.primary.valid_region[
                0
            ] : self._calibration.primary.valid_region[2],
        ]

    def compute_point_cloud(self, block=True) -> Optional[o3d.geometry.PointCloud]:
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
        if block:
            with self._data_condition:
                self._data_condition.wait()
        if self._rgb_frame is None and self._depth is None:
            return None
        if self._compute_point_cloud_on_demand:
            self._update_point_cloud()
        return self._point_cloud

    def compute_im3d(
        self, block=True
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Compute 3D points from the disparity map.

        Parameters
        ----------
        block : bool, optional
            If True, blocks until the next data packet is received. Defaults to True.

        Returns
        -------
        Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]
            A tuple containing the depth map, disparity map, and left frame (if available).
        """
        if block:
            with self._data_condition:
                self._data_condition.wait()
        im3d, disparity, rect = self._3d_packet
        if im3d is None and disparity is None and rect is None:
            return None, None, None
        if self._compute_im3d_on_demand:
            self._update_im3d()
            im3d = self._im3d
        return (
            self._crop_to_valid_primary_region(im3d),
            self._crop_to_valid_primary_region(disparity),
            self._crop_to_valid_primary_region(rect),
        )
