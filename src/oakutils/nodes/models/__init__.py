"""
Functions for creating models for use in the OAK-D pipeline.

Submodules
----------
gaussian
    Module for creating gaussian models.
laplacian
    Module for creating laplacian models.
point_cloud
    Module for creating a point cloud model onboard.
sobel
    Module for creating sobel models.

Functions
---------
create_gaussian
    Creates a gaussian model with a specified kernel size.
create_laplacian
    Creates a laplacian model with a specified kernel size.
create_point_cloud
    Creates a point cloud model onboard.
create_sobel
    Creates a sobel model with a specified kernel size.
create_xyz_matrix
    Use to create a constant reprojection matrix for the given camera matrix and image size.
"""
from . import gaussian, laplacian, point_cloud, sobel
from .gaussian import create_gaussian
from .laplacian import create_laplacian
from .point_cloud import create_point_cloud, create_xyz_matrix
from .sobel import create_sobel

__all__ = [
    "gaussian",
    "laplacian",
    "point_cloud",
    "sobel",
    "create_gaussian",
    "create_laplacian",
    "create_point_cloud",
    "create_xyz_matrix",
    "create_sobel",
]
