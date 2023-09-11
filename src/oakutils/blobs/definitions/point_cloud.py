from __future__ import annotations

from typing import TYPE_CHECKING

from .abstract_model import AbstractModel
from .utils import InputType, ModelType, convert_to_fp16

if TYPE_CHECKING:
    import torch
    from typing_extensions import Self


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
