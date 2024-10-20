# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Models for opening blobs.

Classes
-------
Opening
    nn.Module wrapper for kornia.morphology.opening.
OpeningGray
    nn.Module wrapper for kornia.morphology.opening, with grayscale output.
OpeningBlur
    nn.Module wrapper for kornia.morphology.opening, with gaussian blur.
OpeningBlurGray
    nn.Module wrapper for kornia.morphology.opening, with grayscale output and gaussian blur.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import kornia
import torch

from .abstract_model import AbstractModel
from .utils import InputType, ModelType

if TYPE_CHECKING:
    from typing_extensions import Self


class Opening(AbstractModel):
    """
    nn.Module wrapper for kornia.morphology.opening.

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
            The size of the kernel to use, by default 3

        """
        super().__init__()
        self._kernel = torch.zeros((kernel_size, kernel_size))
        self._kernel[kernel_size // 2, :] = 1.0
        self._kernel[:, kernel_size // 2] = 1.0

    @classmethod
    def model_type(cls: type[Opening]) -> ModelType:
        """
        Use to get the type of input this model takes.

        Returns
        -------
        ModelType
            The type of arguments this model takes.

        """
        return ModelType.KERNEL

    @classmethod
    def input_names(cls: type[Opening]) -> list[tuple[str, InputType]]:
        """
        Use to get the names of the input tensors.

        Returns
        -------
        list[tuple[str, InputType]]
            The names of the input tensors and their datatype.

        """
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: type[Opening]) -> list[str]:
        """
        Use to get the names of the output tensors.

        Returns
        -------
        list[str]
            The names of the output tensors.

        """
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        """
        Use to run the model on the input tensor.

        Parameters
        ----------
        image : torch.Tensor
            The input tensor to run the model on

        Returns
        -------
        torch.Tensor
            The output tensor

        """
        return kornia.morphology.opening(image, self._kernel)


class OpeningGray(AbstractModel):
    """
    nn.Module wrapper for kornia.morphology.opening, with grayscale output.

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
            The size of the kernel to use, by default 3

        """
        super().__init__()
        self._kernel = torch.zeros((kernel_size, kernel_size))
        self._kernel[kernel_size // 2, :] = 1.0
        self._kernel[:, kernel_size // 2] = 1.0

    @classmethod
    def model_type(cls: type[OpeningGray]) -> ModelType:
        """
        Use to get the type of input this model takes.

        Returns
        -------
        ModelType
            The type of arguments this model takes.

        """
        return ModelType.KERNEL

    @classmethod
    def input_names(cls: type[OpeningGray]) -> list[tuple[str, InputType]]:
        """
        Use to get the names of the input tensors.

        Returns
        -------
        list[tuple[str, InputType]]
            The names of the input tensors and their datatype.

        """
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: type[OpeningGray]) -> list[str]:
        """
        Use to get the names of the output tensors.

        Returns
        -------
        list[str]
            The names of the output tensors.

        """
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        """
        Use to run the model on the input tensor.

        Parameters
        ----------
        image : torch.Tensor
            The input tensor to run the model on

        Returns
        -------
        torch.Tensor
            The output tensor

        """
        opening = kornia.morphology.opening(image, self._kernel)
        return kornia.color.bgr_to_grayscale(opening)


class OpeningBlur(AbstractModel):
    """
    nn.Module wrapper for kornia.morphology.opening, with gaussian blur.

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
            The size of the kernel for the gaussian blur, by default 3
        kernel_size2 : int, optional
            The size of the kernel to use, by default 3
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
    def model_type(cls: type[OpeningBlur]) -> ModelType:
        """
        Use to get the type of input this model takes.

        Returns
        -------
        ModelType
            The type of arguments this model takes.

        """
        return ModelType.DUAL_KERNEL

    @classmethod
    def input_names(cls: type[OpeningBlur]) -> list[tuple[str, InputType]]:
        """
        Use to get the names of the input tensors.

        Returns
        -------
        list[tuple[str, InputType]]
            The names of the input tensors and their datatype.

        """
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: type[OpeningBlur]) -> list[str]:
        """
        Use to get the names of the output tensors.

        Returns
        -------
        list[str]
            The names of the output tensors.

        """
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        """
        Use to run the model on the input tensor.

        Parameters
        ----------
        image : torch.Tensor
            The input tensor to run the model on

        Returns
        -------
        torch.Tensor
            The output tensor

        """
        gaussian = kornia.filters.gaussian_blur2d(
            image,
            (self._kernel_size, self._kernel_size),
            (self._sigma, self._sigma),
        )
        return kornia.morphology.opening(gaussian, self._kernel)


class OpeningBlurGray(AbstractModel):
    """
    nn.Module wrapper for kornia.morphology.opening, with gaussian blur, that outputs grayscale.

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
            The size of the kernel for the gaussian blur, by default 3
        kernel_size2 : int, optional
            The size of the kernel to use, by default 3
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
    def model_type(cls: type[OpeningBlurGray]) -> ModelType:
        """
        Use to get the type of input this model takes.

        Returns
        -------
        ModelType
            The type of arguments this model takes.

        """
        return ModelType.DUAL_KERNEL

    @classmethod
    def input_names(cls: type[OpeningBlurGray]) -> list[tuple[str, InputType]]:
        """
        Use to get the names of the input tensors.

        Returns
        -------
        list[tuple[str, InputType]]
            The names of the input tensors and their datatype.

        """
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: type[OpeningBlurGray]) -> list[str]:
        """
        Use to get the names of the output tensors.

        Returns
        -------
        list[str]
            The names of the output tensors.

        """
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        """
        Use to run the model on the input tensor.

        Parameters
        ----------
        image : torch.Tensor
            The input tensor to run the model on

        Returns
        -------
        torch.Tensor
            The output tensor

        """
        gaussian = kornia.filters.gaussian_blur2d(
            image,
            (self._kernel_size, self._kernel_size),
            (self._sigma, self._sigma),
        )
        opening = kornia.morphology.opening(gaussian, self._kernel)
        return kornia.color.bgr_to_grayscale(opening)
