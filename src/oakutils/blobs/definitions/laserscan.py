# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
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

    def __init__(self: Self, width: int = 5, scans: int = 1) -> None:
        """
        Use to create an instance of the model.

        Parameters
        ----------
        width : int, optional
            The width of either side of center to average on, by default 5
        scans : int, optional
            The number of scans to use, by default 1
            Scans are horizontal lines of depth data, each scan
            is sampled from a different row of the depth image, with
            even spacing. A center scan is always generated and is
            always the middle entry in the output scans.

        """
        super().__init__()
        self._width = width
        self._scans = scans

    @classmethod
    def model_type(cls: type[Laserscan]) -> ModelType:
        """Use to get the type of input this model takes."""
        return ModelType.LASERSCAN

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

        # heights to create laserscans at
        heights = [(i + 1) * (height // (self._scans + 1)) for i in range(self._scans)]

        scans = []
        for h_idx in heights:
            # extract the subimage, which is the center ~width*2 pixels horizontally
            sub_image = image[
                :,
                :,
                h_idx - self._width : h_idx + self._width,
                :,
            ]

            # average each column in sub_image into a single value and create a vector
            scans.append(torch.mean(sub_image, dim=2))

        return torch.cat(scans, dim=1)
