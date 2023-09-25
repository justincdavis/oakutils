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
create_point_cloud
    Creates a point cloud model onboard.
create_sobel
    Creates a sobel model with a specified kernel size.
create_xyz_matrix
    Use to create a constant reprojection matrix for the given camera matrix and image size.
"""
from . import gaussian, gftt, harris, hessian, laplacian, point_cloud, sobel
from .gaussian import create_gaussian
from .gftt import create_gftt
from .harris import create_harris
from .hessian import create_hessian
from .laplacian import create_laplacian
from .point_cloud import create_point_cloud, create_xyz_matrix
from .sobel import create_sobel

__all__ = [
    "gaussian",
    "gftt",
    "harris",
    "hessian",
    "laplacian",
    "point_cloud",
    "sobel",
    "create_gaussian",
    "create_gftt",
    "create_harris",
    "create_hessian",
    "create_laplacian",
    "create_point_cloud",
    "create_xyz_matrix",
    "create_sobel",
]
