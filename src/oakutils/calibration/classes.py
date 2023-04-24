from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


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
    Q: np.ndarray
    baseline: float
    primary: Optional[MonoCalibrationData] = None

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
