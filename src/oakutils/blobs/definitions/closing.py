"""
Models for closing blobs.

Classes
-------
Closing
    nn.Module wrapper for kornia.morphology.closing.
ClosingBlur
    nn.Module wrapper for kornia.morphology.closing, with gaussian blur.
ClosingGray
    nn.Module wrapper for kornia.morphology.closing, with grayscale output.
ClosingBlurGray
    nn.Module wrapper for kornia.morphology.closing, with grayscale output and gaussian blur.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import kornia
import torch

from .abstract_model import AbstractModel
from .utils import InputType, ModelType

if TYPE_CHECKING:
    from typing_extensions import Self


class Closing(AbstractModel):
    """
    nn.Module wrapper for kornia.morphology.closing.

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
    def model_type(cls: Closing) -> ModelType:
        """Use to get the type of input this model takes."""
        return ModelType.KERNEL

    @classmethod
    def input_names(cls: Closing) -> list[tuple[str, InputType]]:
        """Use to get the names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: Closing) -> list[str]:
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
        return kornia.morphology.closing(image, self._kernel)


class ClosingGray(AbstractModel):
    """
    nn.Module wrapper for kornia.morphology.closing, with grayscale output.

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
    def model_type(cls: ClosingGray) -> ModelType:
        """Use to get the type of input this model takes."""
        return ModelType.KERNEL

    @classmethod
    def input_names(cls: ClosingGray) -> list[tuple[str, InputType]]:
        """Use to get the names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: ClosingGray) -> list[str]:
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
        closing = kornia.morphology.closing(image, self._kernel)
        return kornia.color.bgr_to_grayscale(closing)


class ClosingBlur(AbstractModel):
    """
    nn.Module wrapper for kornia.morphology.closing, with gaussian blur.

    Methods
    -------
    forward(image: torch.Tensor) -> torch.Tensor
        Use to run the model on the input tensor.
    """

    def __init__(
        self: Self, kernel_size: int = 3, kernel_size2: int = 3, sigma: float = 1.5
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
    def model_type(cls: ClosingBlur) -> ModelType:
        """Use to get the type of input this model takes."""
        return ModelType.DUAL_KERNEL

    @classmethod
    def input_names(cls: ClosingBlur) -> list[tuple[str, InputType]]:
        """Use to get the names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: ClosingBlur) -> list[str]:
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
            image, (self._kernel_size, self._kernel_size), (self._sigma, self._sigma)
        )
        return kornia.morphology.closing(gaussian, self._kernel)


class ClosingBlurGray(AbstractModel):
    """
    nn.Module wrapper for kornia.morphology.closing, with gaussian blur, that outputs grayscale.

    Methods
    -------
    forward(image: torch.Tensor) -> torch.Tensor
        Use to run the model on the input tensor.
    """

    def __init__(
        self: Self, kernel_size: int = 3, kernel_size2: int = 3, sigma: float = 1.5
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
    def model_type(cls: ClosingBlurGray) -> ModelType:
        """Use to get the type of input this model takes."""
        return ModelType.DUAL_KERNEL

    @classmethod
    def input_names(cls: ClosingBlurGray) -> list[tuple[str, InputType]]:
        """Use to get the names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: ClosingBlurGray) -> list[str]:
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
            image, (self._kernel_size, self._kernel_size), (self._sigma, self._sigma)
        )
        closing = kornia.morphology.closing(gaussian, self._kernel)
        return kornia.color.bgr_to_grayscale(closing)
