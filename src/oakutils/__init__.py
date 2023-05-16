from .camera import Camera
from . import (
    calibration,
    point_clouds,
    blobs,
    nodes,
    tools,
)

__all__ = [
    "Camera",
    "calibration",
    "point_clouds",
    "blobs",
    "nodes",
    "tools",
]
__version__ = "0.0.4"

___doc__ = """
oakutils - Python utilities for the OpenCV AI Kit (OAK-D)

This package contains Python utilities for the OpenCV AI Kit (OAK-D) and
related hardware. It is intended to be used with the Luxonis DepthAI API.
Provides easy-to-use classes for working with the OAK-D and doing
common tasks. Also provides easy methods for working with OpenCV and Open3D.
"""
