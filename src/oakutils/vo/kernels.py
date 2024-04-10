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
# ruff: noqa: ARG001
"""
Submodule for the kernels used in visual odometry pipelines.

Functions
---------
bilinear_interpolate_pixel
    Perform bilinear interpolation of the given pixel value.
bilinear_interpolate_pixels
    Perform bilinear interpolation of the given pixel values.
feature_mask
    Mask out disparity values where the disparity is outside the given range.
rigid_body_filter
    Perform a rigid body filter on the given points.
outlier_removal
    Perform outlier removal on the given points using error from estimated transform.

"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from oakutils import _FLAGS

try:
    from numba import jit  # type: ignore[import-untyped]

    _FLAGS.checkjit()
except ImportError:

    def jit(func: Callable, *args: tuple[Any], **kwargs: dict[str, Any]) -> Callable:
        """Jit function for when numba is not installed or not enabled."""
        return func


def _jit_bilinear_interpolate_pixel(
    func: Callable[[np.ndarray, float, float], float],
) -> Callable[[np.ndarray, float, float], float]:
    return jit(func, nopython=True, cache=_FLAGS.JIT_CACHE, fastmath=_FLAGS.JIT_FASTMATH)  # type: ignore[no-any-return]


@_jit_bilinear_interpolate_pixel
def bilinear_interpolate_pixel(img: np.ndarray, x: float, y: float) -> float:
    """
    Perform bilinear interpolation of the given pixel values.

    Parameters
    ----------
    img : np.ndarray
        The image to interpolate.
    x : float
        The x-coordinate to interpolate.
    y : float
        The y-coordinate to interpolate.

    Returns
    -------
    float
        The interpolated pixel value.

    """
    floor_x, floor_y = int(x), int(y)
    p10, p01, p11 = None, None, None
    p00 = img[floor_y, floor_x]
    h, w = img.shape[0:2]
    if floor_x + 1 < w:
        p10 = img[floor_y, floor_x + 1]
        if floor_y + 1 < h:
            p11 = img[floor_y + 1, floor_x + 1]
    if floor_y + 1 < h:
        p01 = img[floor_y + 1, floor_x]
    r_x, r_y, num, den = x - floor_x, y - floor_y, 0.0, 0.0

    if not np.isinf(p00).any():
        num += (1 - r_x) * (1 - r_y) * p00
        den += (1 - r_x) * (1 - r_y)
        # return p00
    if not (p01 is None or np.isinf(p01).any()):
        num += (1 - r_x) * (r_y) * p01
        den += (1 - r_x) * (r_y)
        # return p01
    if not (p10 is None or np.isinf(p10).any()):
        num += (r_x) * (1 - r_y) * p10
        den += (r_x) * (1 - r_y)
        # return p10
    if not (p11 is None or np.isinf(p11).any()):
        num += r_x * r_y * p11
        den += r_x * r_y
        # return p11
    return num / den


def _jit_bilinear_interpolate_pixels(
    func: Callable[[np.ndarray, list[tuple[float, float]]], np.ndarray],
) -> Callable[[np.ndarray, list[tuple[float, float]]], np.ndarray]:
    return jit(func, nopython=True, cache=_FLAGS.JIT_CACHE, fastmath=_FLAGS.JIT_FASTMATH)  # type: ignore[no-any-return]


@_jit_bilinear_interpolate_pixels
def bilinear_interpolate_pixels(
    img: np.ndarray,
    pts: list[tuple[float, float]],
) -> np.ndarray:
    """
    Perform bilinear interpolation of the given pixel values.

    Parameters
    ----------
    img : np.ndarray
        The image to interpolate.
    pts : list[tuple[float, float]]
        The list of points to interpolate.

    Returns
    -------
    list[float]
        The list of interpolated pixel values.

    """
    return np.array([bilinear_interpolate_pixel(img, x, y) for x, y in pts])


def _jit_feature_mask(
    func: Callable[[np.ndarray, int, int], np.ndarray],
) -> Callable[[np.ndarray, int, int], np.ndarray]:
    return jit(func, nopython=True, cache=_FLAGS.JIT_CACHE, fastmath=_FLAGS.JIT_FASTMATH)  # type: ignore[no-any-return]


@_jit_feature_mask
def feature_mask(disparity: np.ndarray, min_disp: int, max_disp: int) -> np.ndarray:
    """
    Mask out disparity values where the disparity is outside the given range.

    Parameters
    ----------
    disparity : np.ndarray
        The disparity image.
    min_disp : int
        The minimum disparity value.
    max_disp : int
        The maximum disparity value.

    Returns
    -------
    np.ndarray
        The disparity mask.

    """
    mask = (disparity >= min_disp) * (disparity <= max_disp)
    return mask.astype(np.uint8) * 255


def _jit_rigid_body_filter(
    func: Callable[
        [
            np.ndarray,
            np.ndarray,
            float,
        ],
        np.ndarray,
    ],
) -> Callable[
    [
        np.ndarray,
        np.ndarray,
        float,
    ],
    np.ndarray,
]:
    return jit(func, nopython=True, cache=_FLAGS.JIT_CACHE, fastmath=_FLAGS.JIT_FASTMATH)  # type: ignore[no-any-return]


@_jit_rigid_body_filter
def rigid_body_filter(
    prev_pts: np.ndarray,
    curr_pts: np.ndarray,
    rigidity_threshold: float = 0.06,
) -> np.ndarray:
    """
    Perform a rigid body filter on the given points.

    Parameters
    ----------
    prev_pts : list[tuple[float, float]]
        The previous points.
    curr_pts : list[tuple[float, float]]
        The current points.
    rigidity_threshold : float, optional
        The rigidity threshold, by default 0.06

    Returns
    -------
    np.ndarray
        The clique or mask of the rigid points.

    """
    # d1-d2 where columns of d1 = pts and rows of d2 = pts
    # result is matrix with entry [i, j] = pts[i] - pts[j]
    dists = np.tile(curr_pts, (len(curr_pts), 1, 1)).transpose((1, 0, 2)) - np.tile(
        curr_pts,
        (len(curr_pts), 1, 1),
    )
    prev_dists = np.tile(prev_pts, (len(curr_pts), 1, 1)).transpose(
        (1, 0, 2),
    ) - np.tile(
        prev_pts,
        (len(curr_pts), 1, 1),
    )
    delta_dist = np.abs(
        np.linalg.norm(dists, axis=2) - np.linalg.norm(prev_dists, axis=2),
    )
    consistency = (np.abs(delta_dist) < rigidity_threshold).astype(int)
    clique: np.ndarray = np.zeros(len(curr_pts), int)
    num_consistent = np.sum(consistency, axis=0)
    max_consistent = np.argmax(num_consistent)
    clique[max_consistent] = 1
    clique_size = 1
    compatible = consistency[max_consistent]
    for _ in range(len(curr_pts)):
        candidates = (compatible - clique).astype(int)
        if np.sum(candidates) == 0:
            break
        selected = np.argmax(num_consistent * candidates)
        clique[selected] = 1
        clique_size += 1
        # leniency = 1 if clique_size > 4 else 0
        leniency = 0
        compatible = (consistency @ clique >= sum(clique) - leniency).astype(int)
    return clique


def _jit_outlier_removal(
    func: Callable[
        [
            np.ndarray,
            np.ndarray,
            np.ndarray,
            float,
        ],
        tuple[
            np.ndarray,
            np.ndarray,
        ],
    ],
) -> Callable[
    [
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
    ],
    tuple[
        np.ndarray,
        np.ndarray,
    ],
]:
    return jit(func, nopython=True, cache=_FLAGS.JIT_CACHE, fastmath=_FLAGS.JIT_FASTMATH)  # type: ignore[no-any-return]


@_jit_outlier_removal
def outlier_removal(
    curr_pts: np.ndarray,
    next_pts: np.ndarray,
    transform: np.ndarray,
    outlier_threshold: float = 0.02,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Perform outlier removal on the given points using error from estimated transform.

    Parameters
    ----------
    curr_pts : np.ndarray
        The current points.
    next_pts : np.ndarray
        The next points.
    transform : np.ndarray
        The estimated transform.
        This should be acquired from cv2.estimateAffine3D.
    outlier_threshold : float, optional
        The outlier threshold, by default 0.02

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The new current and next points after outlier removal.

    """
    transform = np.vstack([transform, [0, 0, 0, 1]])
    h_pts = np.hstack([next_pts, np.array([[1] * len(next_pts)]).transpose()])
    h_prev = np.hstack(
        [curr_pts, np.array([[1] * len(curr_pts)]).transpose()],
    )
    errors = np.array(
        [
            np.linalg.norm(h_pts[i] - transform @ h_prev[i]) / np.linalg.norm(h_pts[i])
            for i in range(len(h_pts))
        ],
    )
    threshold = outlier_threshold + np.median(errors)
    new_curr_pts = curr_pts[errors < threshold]
    new_next_pts = next_pts[errors < threshold]
    return new_curr_pts, new_next_pts
