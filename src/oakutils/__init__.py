from . import blobs, calibration, filters, nodes, point_clouds, tools
from .legacy_camera import Camera as LegacyCamera

__all__ = [
    "LegacyCamera",
    "blobs",
    "calibration",
    "filters",
    "nodes",
    "point_clouds",
    "tools",
]
__version__ = "1.0.1"

___doc__ = """
oakutils - Python utilities for the OpenCV AI Kit (OAK-D)

This package contains Python utilities for the OpenCV AI Kit (OAK-D) and
related hardware. It is intended to be used with the Luxonis DepthAI API.
Provides easy-to-use classes for working with the OAK-D and doing
common tasks. Also provides easy methods for working with OpenCV and Open3D.
"""
