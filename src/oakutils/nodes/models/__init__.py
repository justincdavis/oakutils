# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Functions for creating models for use in the OAK-D pipeline.

Submodules
----------
gaussian
    Module for creating gaussian models.
gftt
    Module for creating gftt models.
harris
    Module for creating harris models.
hessian
    Module for creating hessian models.
laplacian
    Module for creating laplacian models.
laserscan
    Module for creating laserscan models.
point_cloud
    Module for creating a point cloud model onboard.
sobel
    Module for creating sobel models.

Functions
---------
create_gaussian
    Creates a gaussian model with a specified kernel size.
create_gftt
    Creates a gftt model.
create_harris
    Creates a harris model.
create_hessian
    Creates a hessian model.
create_laplacian
    Creates a laplacian model with a specified kernel size.
create_laserscan
    Creates a laserscan model with a specified width.
create_point_cloud
    Creates a point cloud model onboard.
create_sobel
    Creates a sobel model with a specified kernel size.
create_xyz_matrix
    Use to create a constant reprojection matrix for the given camera matrix and image size.
get_laserscan
    Use to get the laserscan data from the laserscan model.
get_point_cloud_buffer
    Use to get the point cloud buffer from the point cloud model.
"""

from __future__ import annotations

import logging

from . import gaussian, gftt, harris, hessian, laplacian, point_cloud, sobel
from .gaussian import create_gaussian
from .gftt import create_gftt
from .harris import create_harris
from .hessian import create_hessian
from .laplacian import create_laplacian
from .laserscan import create_laserscan, get_laserscan
from .point_cloud import create_point_cloud, create_xyz_matrix, get_point_cloud_buffer
from .sobel import create_sobel

_log = logging.getLogger(__name__)

__all__ = [
    "create_gaussian",
    "create_gftt",
    "create_harris",
    "create_hessian",
    "create_laplacian",
    "create_laserscan",
    "create_point_cloud",
    "create_sobel",
    "create_xyz_matrix",
    "gaussian",
    "get_laserscan",
    "get_point_cloud_buffer",
    "gftt",
    "harris",
    "hessian",
    "laplacian",
    "point_cloud",
    "sobel",
]

_log.debug("Loaded nodes.models")
