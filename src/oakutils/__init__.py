# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
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
benchmark
    Contains utilities for benchmarking characteristics of OAK devices.
blobs
    Contains utilities for working with blobs.
calibration
    Contains utilities for working with calibration.
core
    Contains core utilities for working with the OAK-D.
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

Functions
---------
set_log_level
    Set the log level for the oakutils package.
create_device
    Create a DepthAI device object from a pipeline.

"""

from __future__ import annotations

# setup the logger before importing anything else
import contextlib
import logging
import os
import sys


# Created from answer by Dennis at:
# https://stackoverflow.com/questions/7621897/python-logging-module-globally
def _setup_logger(level: str | None = None) -> None:
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
    try:
        log_level = level_map[level]
    except KeyError:
        log_level = logging.WARNING

    # create logger
    logger = logging.getLogger(__package__)
    logger.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)


def set_log_level(level: str) -> None:
    """
    Set the log level for the oakutils package.

    Parameters
    ----------
    level : str
        The log level to set. One of "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".

    Raises
    ------
    ValueError
        If the level is not one of the allowed values.

    """
    if level.upper() not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        err_msg = f"Invalid log level: {level}"
        raise ValueError(err_msg)
    _setup_logger(level)


level = os.getenv("OAKUTILS_LOG_LEVEL")
_setup_logger(level)
_log = logging.getLogger(__name__)
if level is not None and level.upper() not in [
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
]:
    _log.warning(f"Invalid log level: {level}. Using default log level: WARNING")

# handle imports which must occur first
from . import core

if "create_device" not in dir(core):
    err_msg = "The 'core' module must contain a 'create_device' function."
    raise ImportError(err_msg)

# handle other imports
from . import (
    aruco,
    benchmark,
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
from ._webcam import Webcam
from .core import create_device
from .vpu import VPU

__all__ = [
    "VPU",
    "ApiCamera",
    "Webcam",
    "aruco",
    "benchmark",
    "blobs",
    "calibration",
    "core",
    "create_device",
    "filters",
    "nodes",
    "optimizer",
    "point_clouds",
    "set_log_level",
    "tools",
    "vpu",
]

with contextlib.suppress(ImportError):
    from ._legacy_camera import LegacyCamera

    __all__ += ["LegacyCamera"]

__version__ = "1.5.3"

_log.info(f"Initialized oakutils with version {__version__}")
