from typing import Optional, Tuple
import math

import cv2
import numpy as np

from ..calibration import StereoCalibrationData


class WLSFilter:
    """
    A class for computing the weighted-least-squares filter,
    on disparity images.
    """

    def __init__(
        self,
        cam_data: StereoCalibrationData,
        l: int = 8000,
        s: float = 1.0,
        disp_levels: int = 96,
    ):
        """
        Creates a WLSFilter object.

        Parameters
        ----------
        cam_data : StereoCalibrationData
            The stereo calibration data.
        l : int
            The lambda parameter for the WLS filter. Defaults to 8000.
        s : float
            The sigma parameter for the WLS filter. Defaults to 1.0.
        disp_levels : int
            The number of disparity levels in the matcher. Defaults to 96.
        """
        self._data: StereoCalibrationData = cam_data
        self._lambda: int = l
        self._sigma: float = s
        self._disp_levels: int = disp_levels
        self._depth_scale_left: Optional[float] = None
        self._depth_scale_right: Optional[float] = None
        self._filter = cv2.ximgproc.createDisparityWLSFilterGeneric(False)
        self._filter.setLambda(self._lambda)
        self._filter.setSigmaColor(self._sigma)

    def filter(
        self, disparity: np.ndarray, mono_frame: np.ndarray, use_mono_left: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filters the disparity image.

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

        filtered_disp = self._filter.filter(disparity, mono_frame)
        with np.errstate(divide="ignore"):
            depth = (depth_scale / filtered_disp).astype(np.uint16)

        return filtered_disp, depth
