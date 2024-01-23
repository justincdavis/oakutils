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
Module for tools related to calculating spatials locations.

Classes
-------
HostSpatialsCalc
    Class for calculating spatial coordinates on the host.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    import depthai as dai
    from typing_extensions import Self

    from oakutils.calibration import CalibrationData


class HostSpatialsCalc:
    """
    Class for calculating spatial coordinates on the host.

    Attributes
    ----------
    delta : int
        The delta parameter for the spatial coordinates calculation.
        Determines how many neighboring pixels to include in the calculation.
    thresh_low : int
        The lower threshold for the spatial coordinates calculation.
    thresh_high : int
        The upper threshold for the spatial coordinates calculation.

    Methods
    -------
    calc_spatials(depth_frame: np.ndarray) -> np.ndarray
        Calculates the spatial coordinates for the given depth frame.

    References
    ----------
    https://github.com/luxonis/depthai-experiments/blob/master/gen2-calc-spatials-on-host/calc.py
    """

    def __init__(
        self: Self,
        data: CalibrationData,
        delta: int = 5,
        thresh_low: int = 200,
        thresh_high: int = 30000,
    ) -> None:
        """
        Use to create a HostSpatialsCalc object.

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
        self._mid_w: int | None = None
        self._mid_h: int | None = None
        self._f_mid_w: float | None = None
        self._f_mid_h: float | None = None
        self._HFOV: float | None = None
        self._i_HFOV: float | None = None
        self._i_angle: float | None = None

    @property
    def delta(self: Self) -> int:
        """The delta parameter for the spatial coordinates calculation."""
        return self._delta

    @delta.setter
    def delta(self: Self, value: int) -> None:
        self._delta = value

    @property
    def thresh_low(self: Self) -> int:
        """The lower threshold for the spatial coordinates calculation."""
        return self._thresh_low

    @thresh_low.setter
    def thresh_low(self: Self, value: int) -> None:
        self._thresh_low = value

    @property
    def thresh_high(self: Self) -> int:
        """The upper threshold for the spatial coordinates calculation."""
        return self._thresh_high

    @thresh_high.setter
    def thresh_high(self: Self, value: int) -> None:
        self._thresh_high = value

    def _check_input(
        self: Self,
        roi: tuple[int, int] | tuple[int, int, int, int],
        frame: np.ndarray,
    ) -> tuple[int, int, int, int]:
        """Use to check if the input is valid, and constrains to the frame size."""
        xywh_size = 4
        if len(roi) == xywh_size:  # xywh
            return roi  # type: ignore[return-value]
        xy_size = 2
        if len(roi) != xy_size:  # not xy or xywh
            err_msg = "You have to pass either ROI (4 values) or point (2 values)!"
            raise ValueError(
                err_msg,
            )
        x = min(max(roi[0], self._delta), frame.shape[1] - self._delta)
        y = min(max(roi[1], self._delta), frame.shape[0] - self._delta)
        return (x - self._delta, y - self._delta, x + self._delta, y + self._delta)

    def calc_spatials(
        self: Self,
        depth_data: dai.ImgFrame,
        roi: tuple[int, int] | tuple[int, int, int, int],
        averaging_method: Callable = np.mean,
    ) -> tuple[float, float, float, tuple[int, int]]:
        """
        Use to compute spatial coordinates of the ROI.

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

        Raises
        ------
        RuntimeError
            If the initialization failed. Should never occur.
        """
        depth_frame: np.ndarray = depth_data.getFrame()

        if self._first_run:
            self._mid_w = int(depth_frame.shape[1] / 2)  # middle of the depth img width
            self._mid_h = int(
                depth_frame.shape[0] / 2,
            )  # middle of the depth img height
            self._f_mid_w = depth_frame.shape[1] / 2.0  # middle of the depth img width
            self._f_mid_h = depth_frame.shape[0] / 2.0  # middle of the depth img height

            # Required information for calculating spatial coordinates on the host
            cam_num: int = depth_data.getInstanceNum()
            if cam_num == 0:
                self._HFOV = self._data.rgb.fov_rad
            elif cam_num == 1:
                self._HFOV = self._data.left.fov_rad
            else:
                self._HFOV = self._data.right.fov_rad

            # angle calc stuff
            self._i_HFOV = math.tan(self._HFOV / 2.0)
            self._i_angle = self._i_HFOV / self._f_mid_w  # type: ignore[operator]

            # reset flag
            self._first_run = False

        roi = self._check_input(
            roi,
            depth_frame,
        )  # If point was passed, convert it to ROI
        xmin, ymin, xmax, ymax = roi

        # Calculate the average depth in the ROI.
        depth_roi = depth_frame[ymin:ymax, xmin:xmax]
        in_range = (self._thresh_low <= depth_roi) & (depth_roi <= self._thresh_high)

        avg_depth = averaging_method(depth_roi[in_range])

        centroid = (  # Get centroid of the ROI
            int((xmax + xmin) / 2),
            int((ymax + ymin) / 2),
        )

        # assert self._mid_w, self._mid_h are not None
        if self._mid_w is None or self._mid_h is None:
            err_msg = "self._mid_w or self._mid_h is None, initialization error"
            raise RuntimeError(
                err_msg,
            )

        bb_x_pos = centroid[0] - self._mid_w
        bb_y_pos = centroid[1] - self._mid_h

        # asssert self._i_angle is not None
        if self._i_angle is None:
            err_msg = "self._i_angle is None, initialization error"
            raise RuntimeError(err_msg)

        angle_x = math.atan(self._i_angle * bb_x_pos)
        angle_y = math.atan(self._i_angle * bb_y_pos)

        spatials = (
            avg_depth,
            avg_depth * math.tan(angle_x),
            -avg_depth * math.tan(angle_y),
        )
        return *spatials, centroid
