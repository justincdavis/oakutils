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
Module for dilation models.

Classes
-------
Dilation
    nn.Module wrapper for kornia.morphology.dilation.
DilationBlur
    nn.Module wrapper for kornia.morphology.dilation, with gaussian blur.
DilationGray
    nn.Module wrapper for kornia.morphology.dilation, with grayscale output.
DilationBlurGray
    nn.Module wrapper for kornia.morphology.dilation, with grayscale output and gaussian blur.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import kornia
import torch

from .abstract_model import AbstractModel
from .utils import InputType, ModelType

if TYPE_CHECKING:
    from typing_extensions import Self


class Dilation(AbstractModel):
    """
    nn.Module wrapper for kornia.morphology.dilation.

    Methods
    -------
    forward(image: torch.Tensor) -> torch.Tensor
        Use to run the model on the input tensor.
    """

    def __init__(self: Self, kernel_size: int = 3) -> None:
        """
        Use to create an instance of the model.

        Parameters
        ----------
        kernel_size : int, optional
            The size of the kernel for the gaussian blur, by default 3
        """
        super().__init__()
        self._kernel = torch.zeros((kernel_size, kernel_size))
        self._kernel[kernel_size // 2, :] = 1.0
        self._kernel[:, kernel_size // 2] = 1.0

    @classmethod
    def model_type(cls: type[Dilation]) -> ModelType:
        """Use to get the type of input this model takes."""
        return ModelType.KERNEL

    @classmethod
    def input_names(cls: type[Dilation]) -> list[tuple[str, InputType]]:
        """Use to get the names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: type[Dilation]) -> list[str]:
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
        return kornia.morphology.dilation(image, self._kernel)


class DilationGray(AbstractModel):
    """
    nn.Module wrapper for kornia.morphology.dilation, with grayscale output.

    Methods
    -------
    forward(image: torch.Tensor) -> torch.Tensor
        Use to run the model on the input tensor.
    """

    def __init__(self: Self, kernel_size: int = 3) -> None:
        """
        Use to create an instance of the model.

        Parameters
        ----------
        kernel_size : int, optional
            The size of the kernel for the gaussian blur, by default 3
        """
        super().__init__()
        self._kernel = torch.zeros((kernel_size, kernel_size))
        self._kernel[kernel_size // 2, :] = 1.0
        self._kernel[:, kernel_size // 2] = 1.0

    @classmethod
    def model_type(cls: type[DilationGray]) -> ModelType:
        """Use to get the type of input this model takes."""
        return ModelType.KERNEL

    @classmethod
    def input_names(cls: type[DilationGray]) -> list[tuple[str, InputType]]:
        """Use to get the names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: type[DilationGray]) -> list[str]:
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
        dilation = kornia.morphology.dilation(image, self._kernel)
        return kornia.color.bgr_to_grayscale(dilation)


class DilationBlur(AbstractModel):
    """
    nn.Module wrapper for kornia.morphology.dilation, with gaussian blur.

    Methods
    -------
    forward(image: torch.Tensor) -> torch.Tensor
        Use to run the model on the input tensor.
    """

    def __init__(
        self: Self,
        kernel_size: int = 3,
        kernel_size2: int = 3,
        sigma: float = 1.5,
    ) -> None:
        """
        Use to create an instance of the model.

        Parameters
        ----------
        kernel_size : int, optional
            The size of the kernel to use, by default 3
        kernel_size2 : int, optional
            The size of the second kernel to use, by default 3
        sigma : float, optional
            The sigma value for the gaussian blur, by default 1.5
        """
        super().__init__()
        self._kernel = torch.zeros((kernel_size2, kernel_size2))
        self._kernel[kernel_size // 2, :] = 1.0
        self._kernel[:, kernel_size // 2] = 1.0
        self._kernel_size = kernel_size
        self._sigma = sigma

    @classmethod
    def model_type(cls: type[DilationBlur]) -> ModelType:
        """Use to get the type of input this model takes."""
        return ModelType.DUAL_KERNEL

    @classmethod
    def input_names(cls: type[DilationBlur]) -> list[tuple[str, InputType]]:
        """Use to get the names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: type[DilationBlur]) -> list[str]:
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
        gaussian = kornia.filters.gaussian_blur2d(
            image,
            (self._kernel_size, self._kernel_size),
            (self._sigma, self._sigma),
        )
        return kornia.morphology.dilation(gaussian, self._kernel)


class DilationBlurGray(AbstractModel):
    """
    nn.Module wrapper for kornia.morphology.dilation, with gaussian blur, that outputs grayscale.

    Methods
    -------
    forward(image: torch.Tensor) -> torch.Tensor
        Use to run the model on the input tensor.
    """

    def __init__(
        self: Self,
        kernel_size: int = 3,
        kernel_size2: int = 3,
        sigma: float = 1.5,
    ) -> None:
        """
        Use to create an instance of the model.

        Parameters
        ----------
        kernel_size : int, optional
            The size of the kernel to use, by default 3
        kernel_size2 : int, optional
            The size of the second kernel to use, by default 3
        sigma : float, optional
            The sigma value for the gaussian blur, by default 1.5
        """
        super().__init__()
        self._kernel = torch.zeros((kernel_size2, kernel_size2))
        self._kernel[kernel_size // 2, :] = 1.0
        self._kernel[:, kernel_size // 2] = 1.0
        self._kernel_size = kernel_size
        self._sigma = sigma

    @classmethod
    def model_type(cls: type[DilationBlurGray]) -> ModelType:
        """Use to get the type of input this model takes."""
        return ModelType.DUAL_KERNEL

    @classmethod
    def input_names(cls: type[DilationBlurGray]) -> list[tuple[str, InputType]]:
        """Use to get the names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: type[DilationBlurGray]) -> list[str]:
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
        gaussian = kornia.filters.gaussian_blur2d(
            image,
            (self._kernel_size, self._kernel_size),
            (self._sigma, self._sigma),
        )
        dilation = kornia.morphology.dilation(gaussian, self._kernel)
        return kornia.color.bgr_to_grayscale(dilation)
