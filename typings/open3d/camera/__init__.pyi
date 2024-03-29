"""
This type stub file was generated by pyright.
"""

import numpy as np
from typing import Any, ClassVar, List, overload

"""
This type stub file was generated by pyright.
"""
Kinect2ColorCameraDefault: PinholeCameraIntrinsicParameters
Kinect2DepthCameraDefault: PinholeCameraIntrinsicParameters
PrimeSenseDefault: PinholeCameraIntrinsicParameters
class PinholeCameraIntrinsic:
    height: int
    intrinsic_matrix: numpy.ndarray[numpy.float64[3, 3]]
    width: int
    @overload
    def __init__(self) -> Any:
        ...
    
    @overload
    def __init__(self, arg0) -> Any:
        ...
    
    @overload
    def __init__(self, width, height, intrinsic_matrix) -> Any:
        ...
    
    @overload
    def __init__(self, width, height, fx, fy, cx, cy) -> Any:
        ...
    
    @overload
    def __init__(self, param) -> Any:
        ...
    
    def get_focal_length(self) -> Any:
        ...
    
    def get_principal_point(self) -> Any:
        ...
    
    def get_skew(self) -> Any:
        ...
    
    def is_valid(self) -> Any:
        ...
    
    def set_intrinsics(self, width, height, fx, fy, cx, cy) -> Any:
        ...
    
    def __copy__(self) -> PinholeCameraIntrinsic:
        ...
    
    def __deepcopy__(self, arg0: dict) -> PinholeCameraIntrinsic:
        ...
    


class PinholeCameraIntrinsicParameters:
    __members__: ClassVar[dict] = ...
    Kinect2ColorCameraDefault: ClassVar[PinholeCameraIntrinsicParameters] = ...
    Kinect2DepthCameraDefault: ClassVar[PinholeCameraIntrinsicParameters] = ...
    PrimeSenseDefault: ClassVar[PinholeCameraIntrinsicParameters] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None:
        ...
    
    def __eq__(self, other: object) -> bool:
        ...
    
    def __ge__(self, other: object) -> bool:
        ...
    
    def __gt__(self, other: object) -> bool:
        ...
    
    def __hash__(self) -> int:
        ...
    
    def __index__(self) -> int:
        ...
    
    def __int__(self) -> int:
        ...
    
    def __le__(self, other: object) -> bool:
        ...
    
    def __lt__(self, other: object) -> bool:
        ...
    
    def __ne__(self, other: object) -> bool:
        ...
    
    @property
    def name(self) -> str:
        ...
    
    @property
    def value(self) -> int:
        ...
    


class PinholeCameraParameters:
    extrinsic: numpy.ndarray[numpy.float64[4, 4]]
    intrinsic: PinholeCameraIntrinsic
    @overload
    def __init__(self) -> None:
        ...
    
    @overload
    def __init__(self, arg0: PinholeCameraParameters) -> None:
        ...
    
    def __copy__(self) -> PinholeCameraParameters:
        ...
    
    def __deepcopy__(self, arg0: dict) -> PinholeCameraParameters:
        ...
    


class PinholeCameraTrajectory:
    parameters: List[PinholeCameraParameters]
    @overload
    def __init__(self) -> None:
        ...
    
    @overload
    def __init__(self, arg0: PinholeCameraTrajectory) -> None:
        ...
    
    def __copy__(self) -> PinholeCameraTrajectory:
        ...
    
    def __deepcopy__(self, arg0: dict) -> PinholeCameraTrajectory:
        ...
    


