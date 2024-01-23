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
Model definitions for GFTT response models.

Classes
-------
GFTT
    nn.Module wrapper for kornia.feature.GFTT.
GFTTBlur
    nn.Module wrapper for kornia.feature.GFTT_response(kornia.filters.gaussian_blur2d).
GFTTGray
    nn.Module wrapper for kornia.feature.GFTT_response, with grayscale output.
GFTTBlurGray
    nn.Module wrapper for kornia.feature.GFTT_response(kornia.filters.gaussian_blur2d),
      with grayscale output.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import kornia

from .abstract_model import AbstractModel
from .utils import InputType, ModelType

if TYPE_CHECKING:
    import torch
    from typing_extensions import Self


class GFTT(AbstractModel):
    """
    nn.Module wrapper for kornia.feature.GFTT.

    Methods
    -------
    forward(image: torch.Tensor) -> torch.Tensor
        Use to run the model on the input tensor.
    """

    def __init__(self: Self) -> None:
        """Use to create an instance of the model."""
        super().__init__()

    @classmethod
    def model_type(cls: type[GFTT]) -> ModelType:
        """Use to get the type of input this model takes."""
        return ModelType.NONE

    @classmethod
    def input_names(cls: type[GFTT]) -> list[tuple[str, InputType]]:
        """Use to get the names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: type[GFTT]) -> list[str]:
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
        return kornia.feature.gftt_response(image)


class GFTTBlur(AbstractModel):
    """
    nn.Module wrapper for kornia.feature.GFTT_response(kornia.filters.gaussian_blur2d).

    Methods
    -------
    forward(image: torch.Tensor) -> torch.Tensor
        Use to run the model on the input tensor.
    """

    def __init__(self: Self, kernel_size: int = 3, sigma: float = 1.5) -> None:
        """
        Use to create an instance of the model.

        Parameters
        ----------
        kernel_size : int, optional
            The size of the kernel for the gaussian blur, by default 3
        sigma : float, optional
            The sigma value for the gaussian blur, by default 1.5
        """
        super().__init__()
        self._kernel_size = kernel_size
        self._sigma = sigma

    @classmethod
    def model_type(cls: type[GFTTBlur]) -> ModelType:
        """Use to get the type of input this model takes."""
        return ModelType.KERNEL

    @classmethod
    def input_names(cls: type[GFTTBlur]) -> list[tuple[str, InputType]]:
        """Use to get the names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: type[GFTTBlur]) -> list[str]:
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
        return kornia.feature.gftt_response(
            kornia.filters.gaussian_blur2d(
                image,
                (self._kernel_size, self._kernel_size),
                (self._sigma, self._sigma),
            ),
        )


class GFTTGray(AbstractModel):
    """
    nn.Module wrapper for kornia.feature.GFTT_response, with grayscale output.

    Methods
    -------
    forward(image: torch.Tensor) -> torch.Tensor
        Use to run the model on the input tensor.
    """

    def __init__(self: Self) -> None:
        """Use to create an instance of the model."""
        super().__init__()

    @classmethod
    def model_type(cls: type[GFTTGray]) -> ModelType:
        """Use to get the type of input this model takes."""
        return ModelType.NONE

    @classmethod
    def input_names(cls: type[GFTTGray]) -> list[tuple[str, InputType]]:
        """Use to get the names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: type[GFTTGray]) -> list[str]:
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
        gftt = kornia.feature.gftt_response(image)
        return kornia.color.bgr_to_grayscale(gftt)


class GFTTBlurGray(AbstractModel):
    """
    nn.Module wrapper for kornia.feature.GFTT_response(kornia.filters.gaussian_blur2d), with grayscale output.

    Methods
    -------
    forward(image: torch.Tensor) -> torch.Tensor
        Use to run the model on the input tensor.
    """

    def __init__(self: Self, kernel_size: int = 3, sigma: float = 1.5) -> None:
        """
        Use to create an instance of the model.

        Parameters
        ----------
        kernel_size : int, optional
            The size of the kernel for the gaussian blur, by default 3
        sigma : float, optional
            The sigma value for the gaussian blur, by default 1.5
        """
        super().__init__()
        self._kernel_size = kernel_size
        self._sigma = sigma

    @classmethod
    def model_type(cls: type[GFTTBlurGray]) -> ModelType:
        """Use to get the type of input this model takes."""
        return ModelType.KERNEL

    @classmethod
    def input_names(cls: type[GFTTBlurGray]) -> list[tuple[str, InputType]]:
        """Use to get the names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: type[GFTTBlurGray]) -> list[str]:
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
        gftt = kornia.feature.gftt_response(
            kornia.filters.gaussian_blur2d(
                image,
                (self._kernel_size, self._kernel_size),
                (self._sigma, self._sigma),
            ),
        )
        return kornia.color.bgr_to_grayscale(gftt)
