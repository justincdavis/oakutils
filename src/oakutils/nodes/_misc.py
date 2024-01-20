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
from __future__ import annotations

import numpy as np


# https://docs.luxonis.com/projects/api/en/latest/samples/Yolo/tiny_yolo/#rgb-tiny-yolo
def frame_norm(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    """
    Use to adjust a bounding box returned from a YoloDetectionModel node.

    Parameters
    ----------
    frame : np.ndarray
        The frame to adjust the bounding box for
    bbox : tuple[float, float, float, float]
        The bounding box to adjust


    Returns
    -------
    tuple[int, int, int, int]
        The adjusted bounding box

    References
    ----------
    https://docs.luxonis.com/projects/api/en/latest/samples/Yolo/tiny_yolo/#rgb-tiny-yolo
    """
    norm_vals: np.ndarray = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    int_bbox: tuple[int, ...] = tuple(
        (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int),
    )
    return int_bbox  # type: ignore[return-value]
