"""
Package for Python utilities for the OpenCV AI Kit (OAK-D) and related hardware.

This package contains Python utilities for the OpenCV AI Kit (OAK-D) and
related hardware. It is intended to be used with the Luxonis DepthAI API or SDK.
Provides easy-to-use classes for working with the OAK-D and doing
common tasks. Also provides easy methods for working with OpenCV and Open3D.

Submodules
----------
aruco
    Contains utilities for working with ArUco markers.
blobs
    Contains utilities for working with blobs.
calibration
    Contains utilities for working with calibration.
filters
    Contains utilities for working with filters.
nodes
    Contains utilities for creating nodes.
point_clouds
    Contains utilities for working with point clouds.
tools
    Contains general tools and utilities.

Classes
-------
ApiCamera
    A lightweight class for creating custom pipelines using callbacks.
LegacyCamera
    A class for using the color, mono, and imu sensors on the OAK-D.
Webcam
    A class for reading frames from an OAK using the same interface as cv2.VideoCapture.
"""
from . import aruco, blobs, calibration, filters, nodes, point_clouds, tools
from .api_camera import Camera as ApiCamera
from .legacy_camera import Camera as LegacyCamera
from .webcam import Webcam

__all__ = [
    "ApiCamera",
    "LegacyCamera",
    "Webcam",
    "aruco",
    "blobs",
    "calibration",
    "filters",
    "nodes",
    "point_clouds",
    "tools",
]
__version__ = "1.1.0"

___doc__ = """
oakutils - Python utilities for the OpenCV AI Kit (OAK-D)

This package contains Python utilities for the OpenCV AI Kit (OAK-D) and
related hardware. It is intended to be used with the Luxonis DepthAI API.
Provides easy-to-use classes for working with the OAK-D and doing
common tasks. Also provides easy methods for working with OpenCV and Open3D.
"""
