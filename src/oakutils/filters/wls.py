# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Module for using the WLS filter on disparity images.

Classes
-------
WLSFilter
    A class for computing the weighted-least-squares filter,
    on disparity images.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cv2  # type: ignore[import]
import numpy as np

if TYPE_CHECKING:
    from typing_extensions import Self

    from oakutils.calibration import StereoCalibrationData


class WLSFilter:
    """A class for computing the weighted-least-squares filter on disparity images."""

    def __init__(
        self: Self,
        cam_data: StereoCalibrationData,
        lamb: int = 8000,
        sigma: float = 1.0,
        disp_levels: int = 96,
    ) -> None:
        """
        Use to create a WLSFilter object.

        Parameters
        ----------
        cam_data : StereoCalibrationData
            The stereo calibration data.
        lamb : int
            The lambda parameter for the WLS filter. Defaults to 8000.
        sigma : float
            The sigma parameter for the WLS filter. Defaults to 1.0.
        disp_levels : int
            The number of disparity levels in the matcher. Defaults to 96.

        """
        self._data: StereoCalibrationData = cam_data
        self._lambda: int = lamb
        self._sigma: float = sigma
        self._disp_levels: int = disp_levels
        self._depth_scale_left: float | None = None
        self._depth_scale_right: float | None = None
        self._filter = cv2.ximgproc.createDisparityWLSFilterGeneric(  # type: ignore[attr-defined]
            use_confidence=False,
        )
        self._filter.setLambda(self._lambda)
        self._filter.setSigmaColor(self._sigma)

    @property
    def lamb(self: Self) -> int:
        """
        The lambda parameter for the WLS filter.

        Returns
        -------
        int
            The lambda parameter.

        """
        return self._lambda

    @lamb.setter
    def lamb(self: Self, value: int) -> None:
        """
        Use to set the lambda parameter for the WLS filter.

        Parameters
        ----------
        value : int
            The new lambda parameter.

        """
        self._lambda = value
        self._filter.setLambda(self._lambda)

    @property
    def sigma(self: Self) -> float:
        """
        The sigma parameter for the WLS filter.

        Returns
        -------
        float
            The sigma parameter.

        """
        return self._sigma

    @sigma.setter
    def sigma(self: Self, value: float) -> None:
        """
        Use to set the sigma parameter for the WLS filter.

        Parameters
        ----------
        value : float
            The new sigma parameter.

        """
        self._sigma = value
        self._filter.setSigmaColor(self._sigma)

    def filter_frame(
        self: Self,
        disparity: np.ndarray,
        mono_frame: np.ndarray,
        *,
        use_mono_left: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Use to filter the disparity image.

        Parameters
        ----------
        disparity : np.ndarray
            The disparity image to filter.
        mono_frame : np.ndarray
            The mono frame to use for the filter.
        use_mono_left : bool
            Whether to use the left mono frame. Defaults to True.

        Returns
        -------
        np.ndarray
            The filtered disparity image.
        np.ndarray
            The new depth image.

        """
        if use_mono_left is None:
            use_mono_left = True

        if self._depth_scale_left is None:
            self._depth_scale_left = self._data.baseline * (
                disparity.shape[1]
                / (2.0 * math.tan(math.radians(self._data.left.fov / 2)))
            )
            self._depth_scale_right = self._data.baseline * (
                disparity.shape[1]
                / (2.0 * math.tan(math.radians(self._data.right.fov / 2)))
            )

        depth_scale = (
            self._depth_scale_left if use_mono_left else self._depth_scale_right
        )

        filtered_disp: np.ndarray = self._filter.filter(disparity, mono_frame)
        with np.errstate(divide="ignore"):
            depth: np.ndarray = (depth_scale / filtered_disp).astype(np.uint16)

        return filtered_disp, depth
