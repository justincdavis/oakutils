from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .abstract_model import AbstractModel, InputType, ModelType
from .utils import convert_to_fp16

if TYPE_CHECKING:
    import torch
    from typing_extensions import Self


def create_xyz(width: int, height: int, camera_matrix: np.ndarray) -> np.ndarray:
    """Creates a constant reprojection matrix for the given camera matrix and image size.
    This is for generating the input to the point cloud generation model.

    Parameters
    ----------
    width : int
        The width of the image
    height : int
        The height of the image
    camera_matrix : np.ndarray
        The camera matrix to use for the reprojection
        This should be a 3x3 matrix

    Returns
    -------
    np.ndarray
        The reprojection matrix
    """
    xs = np.linspace(0, width - 1, width, dtype=np.float32)
    ys = np.linspace(0, height - 1, height, dtype=np.float32)

    # generate grid by stacking coordinates
    base_grid = np.stack(np.meshgrid(xs, ys))  # WxHx2
    points_2d = base_grid.transpose(1, 2, 0)  # 1xHxWx2

    # unpack coordinates
    u_coord: np.ndarray = points_2d[..., 0]
    v_coord: np.ndarray = points_2d[..., 1]

    # unpack intrinsics
    fx: np.ndarray = camera_matrix[0, 0]
    fy: np.ndarray = camera_matrix[1, 1]
    cx: np.ndarray = camera_matrix[0, 2]
    cy: np.ndarray = camera_matrix[1, 2]

    # projective
    x_coord: np.ndarray = (u_coord - cx) / fx
    y_coord: np.ndarray = (v_coord - cy) / fy

    xyz = np.stack([x_coord, y_coord], axis=-1)
    return np.pad(xyz, ((0, 0), (0, 0), (0, 1)), "constant", constant_values=1.0)


def _depth_to_3d(depth: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
    # depth should come in Bx1xHxW
    points_depth: torch.Tensor = depth.permute(0, 2, 3, 1)  # 1xHxWx1
    points_3d: torch.Tensor = xyz * points_depth
    return points_3d.permute(0, 3, 1, 2)  # Bx3xHxW


class PointCloud(AbstractModel):
    def __init__(self: Self) -> None:
        super().__init__()

    @classmethod
    def model_type(cls: PointCloud) -> ModelType:
        """The type of input this model takes."""
        return ModelType.NONE

    @classmethod
    def input_names(cls: PointCloud) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("xyz", InputType.XYZ), ("depth", InputType.U8)]

    @classmethod
    def output_names(cls: PointCloud) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self: Self, xyz: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        depth_fp16 = convert_to_fp16(depth)
        return _depth_to_3d(depth_fp16, xyz)
