from __future__ import annotations

import numpy as np


def homogenous_pixel_coord(width: int, height: int) -> np.ndarray:
    """Pixel in homogenous coordinate.

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
    x = np.linspace(0, width - 1, width).astype(int)
    y = np.linspace(0, height - 1, height).astype(int)
    [x, y] = np.meshgrid(x, y)
    return np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))
