# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Model definitions for erosion models.

Classes
-------
Erosion
    nn.Module wrapper for kornia.morphology.erosion.
ErosionBlur
    nn.Module wrapper for kornia.morphology.erosion, with gaussian blur.
ErosionGray
    nn.Module wrapper for kornia.morphology.erosion, with grayscale output.
ErosionBlurGray
    nn.Module wrapper for kornia.morphology.erosion, with grayscale output and gaussian blur.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import kornia
import torch

from .abstract_model import AbstractModel
from .utils import InputType, ModelType

if TYPE_CHECKING:
    from typing_extensions import Self


class Erosion(AbstractModel):
    """
    nn.Module wrapper for kornia.morphology.erosion.

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
    def model_type(cls: type[Erosion]) -> ModelType:
        """
        Use to get the type of input this model takes.

        Returns
        -------
        ModelType
            The type of arguments this model takes.

        """
        return ModelType.KERNEL

    @classmethod
    def input_names(cls: type[Erosion]) -> list[tuple[str, InputType]]:
        """
        Use to get the names of the input tensors.

        Returns
        -------
        list[tuple[str, InputType]]
            The names of the input tensors and their datatype.

        """
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: type[Erosion]) -> list[str]:
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
        return kornia.morphology.erosion(image, self._kernel)


class ErosionGray(AbstractModel):
    """
    nn.Module wrapper for kornia.morphology.erosion, with grayscale output.

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
    def model_type(cls: type[ErosionGray]) -> ModelType:
        """
        Use to get the type of input this model takes.

        Returns
        -------
        ModelType
            The type of arguments this model takes.

        """
        return ModelType.KERNEL

    @classmethod
    def input_names(cls: type[ErosionGray]) -> list[tuple[str, InputType]]:
        """
        Use to get the names of the input tensors.

        Returns
        -------
        list[tuple[str, InputType]]
            The names of the input tensors and their datatype.

        """
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: type[ErosionGray]) -> list[str]:
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
        erosion = kornia.morphology.erosion(image, self._kernel)
        return kornia.color.bgr_to_grayscale(erosion)


class ErosionBlur(AbstractModel):
    """
    nn.Module wrapper for kornia.morphology.erosion, with gaussian blur.

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
    def model_type(cls: type[ErosionBlur]) -> ModelType:
        """
        Use to get the type of input this model takes.

        Returns
        -------
        ModelType
            The type of arguments this model takes.

        """
        return ModelType.DUAL_KERNEL

    @classmethod
    def input_names(cls: type[ErosionBlur]) -> list[tuple[str, InputType]]:
        """
        Use to get the names of the input tensors.

        Returns
        -------
        list[tuple[str, InputType]]
            The names of the input tensors and their datatype.

        """
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: type[ErosionBlur]) -> list[str]:
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
        return kornia.morphology.erosion(gaussian, self._kernel)


class ErosionBlurGray(AbstractModel):
    """
    nn.Module wrapper for kornia.morphology.erosion, with gaussian blur, that outputs grayscale.

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
    def model_type(cls: type[ErosionBlurGray]) -> ModelType:
        """
        Use to get the type of input this model takes.

        Returns
        -------
        ModelType
            The type of arguments this model takes.

        """
        return ModelType.DUAL_KERNEL

    @classmethod
    def input_names(cls: type[ErosionBlurGray]) -> list[tuple[str, InputType]]:
        """
        Use to get the names of the input tensors.

        Returns
        -------
        list[tuple[str, InputType]]
            The names of the input tensors and their datatype.

        """
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: type[ErosionBlurGray]) -> list[str]:
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
        erosion = kornia.morphology.erosion(gaussian, self._kernel)
        return kornia.color.bgr_to_grayscale(erosion)
