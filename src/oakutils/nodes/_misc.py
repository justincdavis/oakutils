# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
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
