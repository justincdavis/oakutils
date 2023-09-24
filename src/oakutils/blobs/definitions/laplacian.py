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

import kornia

from .abstract_model import AbstractModel
from .utils import InputType, ModelType

if TYPE_CHECKING:
    import torch
    from typing_extensions import Self


class Laplacian(AbstractModel):
    """
    nn.Module wrapper for kornia.filters.laplacian.

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
        self._kernel_size = kernel_size

    @classmethod
    def model_type(cls: Laplacian) -> ModelType:
        """Use to get the type of input this model takes."""
        return ModelType.KERNEL

    @classmethod
    def input_names(cls: Laplacian) -> list[tuple[str, InputType]]:
        """Use to get the names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: Laplacian) -> list[str]:
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
        return kornia.filters.laplacian(image, self._kernel_size)


class LaplacianGray(AbstractModel):
    """
    nn.Module wrapper for kornia.filters.laplacian, with grayscale output.

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
        self._kernel_size = kernel_size

    @classmethod
    def model_type(cls: LaplacianGray) -> ModelType:
        """Use to get the type of input this model takes."""
        return ModelType.KERNEL

    @classmethod
    def input_names(cls: LaplacianGray) -> list[tuple[str, InputType]]:
        """Use to get the names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: LaplacianGray) -> list[str]:
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
        laplacian = kornia.filters.laplacian(image, self._kernel_size)
        return kornia.color.bgr_to_grayscale(laplacian)


class LaplacianBlur(AbstractModel):
    """
    nn.Module wrapper for kornia.filters.laplacian, with gaussian blur.

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
            The size of the kernel to use for the gaussian blur, by default 3
        sigma : float, optional
            The sigma value for the gaussian blur, by default 1.5
        """
        super().__init__()
        self._kernel_size = kernel_size
        self._kernel_size2 = kernel_size2
        self._sigma = sigma

    @classmethod
    def model_type(cls: LaplacianBlur) -> ModelType:
        """Use to get the type of input this model takes."""
        return ModelType.DUAL_KERNEL

    @classmethod
    def input_names(cls: LaplacianBlur) -> list[tuple[str, InputType]]:
        """Use to get the names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: LaplacianBlur) -> list[str]:
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
        return kornia.filters.laplacian(gaussian, self._kernel_size2)


class LaplacianBlurGray(AbstractModel):
    """
    nn.Module wrapper for kornia.filters.laplacian, with gaussian blur, that outputs grayscale.

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
            The size of the kernel to use for the gaussian blur, by default 3
        sigma : float, optional
            The sigma value for the gaussian blur, by default 1.5
        """
        super().__init__()
        self._kernel_size = kernel_size
        self._kernel_size2 = kernel_size2
        self._sigma = sigma

    @classmethod
    def model_type(cls: LaplacianBlurGray) -> ModelType:
        """Use to get the type of input this model takes."""
        return ModelType.DUAL_KERNEL

    @classmethod
    def input_names(cls: LaplacianBlurGray) -> list[tuple[str, InputType]]:
        """Use to get the names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: LaplacianBlurGray) -> list[str]:
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
        laplacian = kornia.filters.laplacian(gaussian, self._kernel_size2)
        return kornia.color.bgr_to_grayscale(laplacian)
