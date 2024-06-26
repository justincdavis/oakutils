# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Module for ArUco marker detection and localization.

Classes
-------
ArucoFinder
    Use to find ArUco markers in an image.
ArucoLocalizer
    Use to localize the camera within the world frame using ArUco markers.
ArucoStream
    Used on a video stream to find ArUco markers.
"""

from __future__ import annotations

import logging

from .finder import ArucoFinder
from .localizer import ArucoLocalizer
from .stream import ArucoStream

_log = logging.getLogger(__name__)

__all__ = [
    "ArucoFinder",
    "ArucoLocalizer",
    "ArucoStream",
]

_log.debug("Loaded aruco")
