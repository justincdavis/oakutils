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
Helper functions for creating transformation matrices.

This module contains helper functions for creating transformation matrices.
These matrices can be used to transform points from one coordinate system to
another.

Functions
---------
create_rotation
    Use to create a rotation matrix from a rotation vector.
create_translation
    Use to create a translation vector.
create_transform
    Use to get transformation matrix from a rotation vector and translation vector.
"""
import numpy as np


def create_rotation(theta_x: float, theta_y: float, theta_z: float) -> np.ndarray:
    """
    Use to create a rotation matrix from a rotation vector.

    Parameters
    ----------
    theta_x : float
        The rotation angle around the x axis
    theta_y : float
        The rotation angle around the y axis
    theta_z : float
        The rotation angle around the z axis

    Returns
    -------
    np.ndarray
        The rotation matrix
    """
    rotation_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta_x), -np.sin(theta_x)],
            [0, np.sin(theta_x), np.cos(theta_x)],
        ],
    )

    rotation_y = np.array(
        [
            [np.cos(theta_y), 0, np.sin(theta_y)],
            [0, 1, 0],
            [-np.sin(theta_y), 0, np.cos(theta_y)],
        ],
    )

    rotation_z = np.array(
        [
            [np.cos(theta_z), -np.sin(theta_z), 0],
            [np.sin(theta_z), np.cos(theta_z), 0],
            [0, 0, 1],
        ],
    )

    return rotation_z @ rotation_y @ rotation_x  # type: ignore[no-any-return]


def create_translation(delta_x: float, delta_y: float, delta_z: float) -> np.ndarray:
    """
    Use to create a translation vector.

    Parameters
    ----------
    delta_x : float
        The translation along the x axis
    delta_y : float
        The translation along the y axis
    delta_z : float
        The translation along the z axis

    Returns
    -------
    np.ndarray
        The translation vector
    """
    return np.array([[delta_x, delta_y, delta_z]]).T


def create_transform(
    theta_x: float,
    theta_y: float,
    theta_z: float,
    delta_x: float,
    delta_y: float,
    delta_z: float,
) -> np.ndarray:
    """
    Use to get transformation matrix from a rotation vector and translation vector.

    Parameters
    ----------
    theta_x : float
        The rotation angle around the x axis
    theta_y : float
        The rotation angle around the y axis
    theta_z : float
        The rotation angle around the z axis
    delta_x : float
        The translation along the x axis
    delta_y : float
        The translation along the y axis
    delta_z : float
        The translation along the z axis

    Returns
    -------
    np.ndarray
        The transformation matrix
    """
    rotation_matrix = create_rotation(theta_x, theta_y, theta_z)
    translation_vector = create_translation(delta_x, delta_y, delta_z)
    transform: np.ndarray = np.block(
        [
            [rotation_matrix, translation_vector],
            [0, 0, 0, 1],
        ],  # pyright: ignore[reportArgumentType, reportCallIssue]
    )
    return transform.astype(np.float32)
