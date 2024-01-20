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
Module for pixel coordinate tools.

Functions
---------
homogenous_pixel_coord
    Pixel in homogenous coordinate.
"""
from __future__ import annotations

import numpy as np


def homogenous_pixel_coord(width: int, height: int) -> np.ndarray:
    """
    Pixel in homogenous coordinate.

    Parameters
    ----------
    width : int
        Location of pixel in image width.
    height : int
        Location of pixel in image height.

    Returns
    -------
    np.ndarray
        Pixel coordinate in homogenous coordinate.
        Pixel coordinate: [3, width * height]

    References
    ----------
    https://github.com/luxonis/depthai-experiments/blob/master/gen2-pointcloud/rgbd-pointcloud/utils.py
    """
    x: np.ndarray = np.linspace(0, width - 1, width).astype(int)
    y: np.ndarray = np.linspace(0, height - 1, height).astype(int)
    [x, y] = np.meshgrid(x, y)
    return np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))
