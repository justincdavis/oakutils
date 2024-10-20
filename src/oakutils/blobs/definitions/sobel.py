# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Model definitions for Sobel edge detection.

Classes
-------
Sobel
    nn.Module wrapper for kornia.filters.sobel.
SobelBlur
    nn.Module wrapper for kornia.filters.sobel(kornia.filters.gaussian_blur2d).
SobelGray
    nn.Module wrapper for kornia.filters.sobel, with grayscale output.
SobelBlurGray
    nn.Module wrapper for kornia.filters.sobel(kornia.filters.gaussian_blur2d),
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


class Sobel(AbstractModel):
    """
    nn.Module wrapper for kornia.filters.sobel.

    Methods
    -------
    forward(image: torch.Tensor) -> torch.Tensor
        Use to run the model on the input tensor.

    """

    def __init__(self: Self) -> None:
        """Use to create an instance of the model."""
        super().__init__()

    @classmethod
    def model_type(cls: type[Sobel]) -> ModelType:
        """
        Use to get the type of input this model takes.

        Returns
        -------
        ModelType
            The type of arguments this model takes.

        """
        return ModelType.NONE

    @classmethod
    def input_names(cls: type[Sobel]) -> list[tuple[str, InputType]]:
        """
        Use to get the names of the input tensors.

        Returns
        -------
        list[tuple[str, InputType]]
            The names of the input tensors and their datatype.

        """
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: type[Sobel]) -> list[str]:
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
        return kornia.filters.sobel(image)


class SobelBlur(AbstractModel):
    """
    nn.Module wrapper for kornia.filters.sobel(kornia.filters.gaussian_blur2d).

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
    def model_type(cls: type[SobelBlur]) -> ModelType:
        """
        Use to get the type of input this model takes.

        Returns
        -------
        ModelType
            The type of arguments this model takes.

        """
        return ModelType.KERNEL

    @classmethod
    def input_names(cls: type[SobelBlur]) -> list[tuple[str, InputType]]:
        """
        Use to get the names of the input tensors.

        Returns
        -------
        list[tuple[str, InputType]]
            The names of the input tensors and their datatype.

        """
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: type[SobelBlur]) -> list[str]:
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
        return kornia.filters.sobel(
            kornia.filters.gaussian_blur2d(
                image,
                (self._kernel_size, self._kernel_size),
                (self._sigma, self._sigma),
            ),
        )


class SobelGray(AbstractModel):
    """
    nn.Module wrapper for kornia.filters.sobel, with grayscale output.

    Methods
    -------
    forward(image: torch.Tensor) -> torch.Tensor
        Use to run the model on the input tensor.

    """

    def __init__(self: Self) -> None:
        """Use to create an instance of the model."""
        super().__init__()

    @classmethod
    def model_type(cls: type[SobelGray]) -> ModelType:
        """
        Use to get the type of input this model takes.

        Returns
        -------
        ModelType
            The type of arguments this model takes.

        """
        return ModelType.NONE

    @classmethod
    def input_names(cls: type[SobelGray]) -> list[tuple[str, InputType]]:
        """
        Use to get the names of the input tensors.

        Returns
        -------
        list[tuple[str, InputType]]
            The names of the input tensors and their datatype.

        """
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: type[SobelGray]) -> list[str]:
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
        sobel = kornia.filters.sobel(image)
        return kornia.color.bgr_to_grayscale(sobel)


class SobelBlurGray(AbstractModel):
    """
    nn.Module wrapper for kornia.filters.sobel(kornia.filters.gaussian_blur2d), with grayscale output.

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
    def model_type(cls: type[SobelBlurGray]) -> ModelType:
        """
        Use to get the type of input this model takes.

        Returns
        -------
        ModelType
            The type of arguments this model takes.

        """
        return ModelType.KERNEL

    @classmethod
    def input_names(cls: type[SobelBlurGray]) -> list[tuple[str, InputType]]:
        """
        Use to get the names of the input tensors.

        Returns
        -------
        list[tuple[str, InputType]]
            The names of the input tensors and their datatype.

        """
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: type[SobelBlurGray]) -> list[str]:
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
        sobel = kornia.filters.sobel(
            kornia.filters.gaussian_blur2d(
                image,
                (self._kernel_size, self._kernel_size),
                (self._sigma, self._sigma),
            ),
        )
        return kornia.color.bgr_to_grayscale(sobel)
