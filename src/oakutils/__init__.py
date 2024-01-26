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
# ruff: noqa: E402
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
optimizer
    Contains utilities for optimizing pipelines.
point_clouds
    Contains utilities for working with point clouds.
tools
    Contains general tools and utilities.
vpu
    Contains utilities for using the onboard VPU as a standalone processor.

Classes
-------
ApiCamera
    A lightweight class for creating custom pipelines using callbacks.
LegacyCamera
    A class for using the color, mono, and imu sensors on the OAK-D.
Webcam
    A class for reading frames from an OAK using the same interface as cv2.VideoCapture.
VPU
    A class for using the onboard VPU as a standalone processor.
"""
from __future__ import annotations

# setup the logger before importing anything else
import logging
import os
import sys


# Created from answer by Dennis at:
# https://stackoverflow.com/questions/7621897/python-logging-module-globally
def _setup_logger() -> None:
    # get logging level environment variable
    level = os.getenv("OAKUTILS_LOG_LEVEL")
    if level is not None:
        level = level.upper()
    level_map: dict[str | None, int] = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
        None: logging.WARNING,
    }
    log_level = level_map[level]

    # create logger
    logger = logging.getLogger(__package__)
    logger.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)


_setup_logger()
_log = logging.getLogger(__name__)


from . import (
    aruco,
    blobs,
    calibration,
    filters,
    nodes,
    optimizer,
    point_clouds,
    tools,
    vpu,
)
from ._api_camera import ApiCamera
from ._legacy_camera import LegacyCamera
from ._webcam import Webcam
from .vpu import VPU

__all__ = [
    "VPU",
    "ApiCamera",
    "LegacyCamera",
    "Webcam",
    "aruco",
    "blobs",
    "calibration",
    "filters",
    "nodes",
    "optimizer",
    "point_clouds",
    "tools",
    "vpu",
]
__version__ = "1.4.1"

___doc__ = """
oakutils - Python utilities for the OpenCV AI Kit (OAK-D)

This package contains Python utilities for the OpenCV AI Kit (OAK-D) and
related hardware. It is intended to be used with the Luxonis DepthAI API.
Provides easy-to-use classes for working with the OAK-D and doing
common tasks. Also provides easy methods for working with OpenCV and Open3D.
"""

_log.info(f"Initialized oakutils with version {__version__}")
