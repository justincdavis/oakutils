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
Module for Laplacian models.

Classes
-------
Laplacian
    nn.Module wrapper for kornia.filters.laplacian.
LaplacianGray
    nn.Module wrapper for kornia.filters.laplacian, with grayscale output.
LaplacianBlur
    nn.Module wrapper for kornia.filters.laplacian, with gaussian blur.
LaplacianBlurGray
    nn.Module wrapper for kornia.filters.laplacian, with grayscale output and gaussian blur.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .abstract_model import AbstractModel
from .utils import InputType, ModelType

if TYPE_CHECKING:
    from typing_extensions import Self


class Laserscan(AbstractModel):
    """
    nn.Module for creating a laserscan from a depth image.

    Methods
    -------
    forward(image: torch.Tensor) -> torch.Tensor
        Use to run the model on the input tensor.

    """

    def __init__(self: Self, width: int = 5) -> None:
        """
        Use to create an instance of the model.

        Parameters
        ----------
        width : int, optional
            The width of either side of center to average on, by default 5

        """
        super().__init__()
        self._width = width

    @classmethod
    def model_type(cls: type[Laserscan]) -> ModelType:
        """Use to get the type of input this model takes."""
        return ModelType.WIDTH

    @classmethod
    def input_names(cls: type[Laserscan]) -> list[tuple[str, InputType]]:
        """Use to get the names of the input tensors."""
        return [("input", InputType.U8)]

    @classmethod
    def output_names(cls: type[Laserscan]) -> list[str]:
        """Use to get the names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        """
        Use to run the model on the input tensor.

        Parameters
        ----------
        image : torch.Tensor
            The input tensor to run the model on

        """
        # get the shape of the image
        _, _, height, _ = image.shape
        middle_height = height // 2

        # extract the subimage, which is the center ~width*2 pixels horizontally
        sub_image = image[
            :,
            :,
            middle_height - self._width : middle_height + self._width,
            :,
        ]

        # average each column in sub_image into a single value and create a vector
        return torch.mean(sub_image, dim=2)
