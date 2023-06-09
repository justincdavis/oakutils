from typing import Union, Tuple, Callable, Optional

import math
import numpy as np
import depthai as dai

from ..calibration import CalibrationData


class HostSpatialsCalc:
    """
    Class for calculating spatial coordinates on the host.

    Methods
    -------
    calc_spatials(depth_frame: np.ndarray) -> np.ndarray
        Calculates the spatial coordinates for the given depth frame.

    Attributes
    ----------
    delta : int
        The delta parameter for the spatial coordinates calculation.
        Determines how many neighboring pixels to include in the calculation.
    thresh_low : int
        The lower threshold for the spatial coordinates calculation.
    thresh_high : int
        The upper threshold for the spatial coordinates calculation.

    References
    ----------
    https://github.com/luxonis/depthai-experiments/blob/master/gen2-calc-spatials-on-host/calc.py
    """

    def __init__(
        self,
        data: CalibrationData,
        delta: int = 5,
        thresh_low: int = 200,
        thresh_high: int = 30000,
    ):
        """
        Creates a HostSpatialsCalc object.

        Parameters
        ----------
        data : CalibrationData
            The calibration data.
        delta : int, optional
            The delta parameter for the spatial coordinates calculation.
            Determines how many neighboring pixels to include in the calculation.
            The default is 5.
        thresh_low : int, optional
            The lower threshold for the spatial coordinates calculation.
            The default is 200.
        thresh_high : int, optional
            The upper threshold for the spatial coordinates calculation.
            The default is 30000.
        """
        self._data: CalibrationData = data
        self._delta: int = delta
        self._thresh_low: int = thresh_low  # 20cm
        self._thresh_high: int = thresh_high  # 30m

        # parameters which get resolved on first run
        self._first_run: bool = True
        self._mid_w: Optional[int] = None
        self._mid_h: Optional[int] = None
        self._f_mid_w: Optional[float] = None
        self._f_mid_h: Optional[float] = None
        self._HFOV: Optional[float] = None
        self._i_HFOV: Optional[float] = None
        self._i_angle: Optional[float] = None

    @property
    def delta(self) -> int:
        """
        The delta parameter for the spatial coordinates calculation.
        """
        return self._delta

    @delta.setter
    def delta(self, value: int) -> None:
        self._delta = value

    @property
    def thresh_low(self) -> int:
        """
        The lower threshold for the spatial coordinates calculation.
        """
        return self._thresh_low

    @thresh_low.setter
    def thresh_low(self, value: int) -> None:
        self._thresh_low = value

    @property
    def thresh_high(self) -> int:
        """
        The upper threshold for the spatial coordinates calculation.
        """
        return self._thresh_high

    @thresh_high.setter
    def thresh_high(self, value: int) -> None:
        self._thresh_high = value

    def _check_input(
        self, roi: Union[Tuple[int, int], Tuple[int, int, int, int]], frame: np.ndarray
    ) -> Tuple[int, int, int, int]:
        """
        Checks if the input is valid, and constrains to the frame size.
        """
        if len(roi) == 4:
            return roi
        if len(roi) != 2:
            raise ValueError(
                "You have to pass either ROI (4 values) or point (2 values)!"
            )
        x = min(max(roi[0], self._delta), frame.shape[1] - self._delta)
        y = min(max(roi[1], self._delta), frame.shape[0] - self._delta)
        return (x - self._delta, y - self._delta, x + self._delta, y + self._delta)

    def calc_spatials(
        self,
        depth_data: dai.ImgFrame,
        roi: Union[Tuple[int, int], Tuple[int, int, int, int]],
        averaging_method: Callable = np.mean,
    ) -> Tuple[float, float, float, Tuple[int, int]]:
        """
        Computes spatial coordinates of the ROI.

        Parameters
        ----------
        depth_data : depthai.ImgFrame
            The depth frame and data.
        roi : tuple of int
            The ROI to calculate the spatial coordinates for.
        averaging_method : callable, optional
            The averaging method to use for calculating the depth.
            The default is np.mean.

        Returns
        -------
        float
            The x coordinate of the ROI centroid.
        float
            The y coordinate of the ROI centroid.
        float
            The z coordinate of the ROI centroid.
        Tuple[int, int]
            The centroid of the ROI.
        """

        depth_frame = depth_data.getFrame()

        if self._first_run:
            self._mid_w = int(depth_frame.shape[1] / 2)  # middle of the depth img width
            self._mid_h = int(
                depth_frame.shape[0] / 2
            )  # middle of the depth img height
            self._f_mid_w = depth_frame.shape[1] / 2.0  # middle of the depth img width
            self._f_mid_h = depth_frame.shape[0] / 2.0  # middle of the depth img height

            # Required information for calculating spatial coordinates on the host
            cam_num = depth_data.getInstanceNum()
            if cam_num == 0:
                self._HFOV = self._data.rgb.fov_rad
            elif cam_num == 1:
                self._HFOV = self._data.left.fov_rad
            else:
                self._HFOV = self._data.right.fov_rad

            # angle calc stuff
            self._i_HFOV: float = math.tan(self._HFOV / 2.0)
            self._i_angle: float = self._i_HFOV / self._f_mid_w

            # reset flag
            self._first_run = False

        roi = self._check_input(
            roi, depth_frame
        )  # If point was passed, convert it to ROI
        xmin, ymin, xmax, ymax = roi

        # Calculate the average depth in the ROI.
        depthROI = depth_frame[ymin:ymax, xmin:xmax]
        inRange = (self._thresh_low <= depthROI) & (depthROI <= self._thresh_high)

        averageDepth = averaging_method(depthROI[inRange])

        centroid = (  # Get centroid of the ROI
            int((xmax + xmin) / 2),
            int((ymax + ymin) / 2),
        )

        bb_x_pos = centroid[0] - self._mid_w
        bb_y_pos = centroid[1] - self._mid_h

        angle_x = math.atan(self._i_angle * bb_x_pos)
        angle_y = math.atan(self._i_angle * bb_y_pos)

        spatials = (
            averageDepth,
            averageDepth * math.tan(angle_x),
            -averageDepth * math.tan(angle_y),
        )
        return *spatials, centroid
