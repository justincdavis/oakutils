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
from .finder import ArucoFinder
from .localizer import ArucoLocalizer
from .stream import ArucoStream

__all__ = [
    "ArucoFinder",
    "ArucoLocalizer",
    "ArucoStream",
]
