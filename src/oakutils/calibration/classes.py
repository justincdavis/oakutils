from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import open3d as o3d


@dataclass
class MonoCalibrationData:
    """
    Class to store a mono cameras calibration data
    """

    size: Tuple[int, int]
    K: np.ndarray
    D: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    fov: float
    R: np.ndarray
    T: np.ndarray
    H: np.ndarray
    valid_region: Optional[Tuple[int, int, int, int]] = None
    map_1: Optional[np.ndarray] = None
    map_2: Optional[np.ndarray] = None
    pinhole: Optional[o3d.camera.PinholeCameraIntrinsic] = None


@dataclass
class StereoCalibrationData:
    """
    Class to store a stereo cameras calibration data
    """

    left: MonoCalibrationData
    right: MonoCalibrationData
    R1: np.ndarray
    R2: np.ndarray
    T1: np.ndarray
    T2: np.ndarray
    H_left: np.ndarray
    H_right: np.ndarray
    l2r_extrinsic: np.ndarray
    r2l_extrinsic: np.ndarray
    Q_left: np.ndarray
    Q_right: np.ndarray
    baseline: float
    primary: Optional[MonoCalibrationData] = None
    Q_primary: Optional[np.ndarray] = None
    cv2_Q: Optional[np.ndarray] = None
    R1: Optional[np.ndarray] = None
    R2: Optional[np.ndarray] = None
    P1: Optional[np.ndarray] = None
    P2: Optional[np.ndarray] = None
    valid_region_primary: Optional[Tuple[int, int, int, int]] = None
    pinhole_primary: Optional[o3d.camera.PinholeCameraIntrinsic] = None


@dataclass
class ColorCalibrationData:
    """
    Class to store a color cameras calibration data
    """

    size: Tuple[int, int]
    K: np.ndarray
    D: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    fov: float
    P: Optional[np.ndarray] = None
    valid_region: Optional[Tuple[int, int, int, int]] = None
    map_1: Optional[np.ndarray] = None
    map_2: Optional[np.ndarray] = None
    pinhole: Optional[o3d.camera.PinholeCameraIntrinsic] = None


@dataclass
class CalibrationData:
    """
    Class for rgb and two mono cameras calibration data
    Used for the OAK-D cameras
    """

    rgb: ColorCalibrationData
    left: MonoCalibrationData
    right: MonoCalibrationData
    stereo: StereoCalibrationData
    primary: Optional[MonoCalibrationData] = None
