"""
Module for Hessian blobs models.

Classes
-------
Hessian
    nn.Module wrapper for kornia.feature.Hessian_response.
HessianBlur
    nn.Module wrapper for kornia.feature.Hessian_response, with gaussian blur.
HessianGray
    nn.Module wrapper for kornia.feature.Hessian_response, with grayscale output.
HessianBlurGray
    nn.Module wrapper for kornia.feature.Hessian_response, with grayscale output and gaussian blur.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import kornia

from .abstract_model import AbstractModel
from .utils import InputType, ModelType

if TYPE_CHECKING:
    import torch
    from typing_extensions import Self


class Hessian(AbstractModel):
    """
    nn.Module wrapper for kornia.feature.Hessian.

    Methods
    -------
    forward(image: torch.Tensor) -> torch.Tensor
        Use to run the model on the input tensor.
    """

    def __init__(self: Self) -> None:
        """Use to create an instance of the model."""
        super().__init__()

    @classmethod
    def model_type(cls: Hessian) -> ModelType:
        """Use to get the type of input this model takes."""
        return ModelType.NONE

    @classmethod
    def input_names(cls: Hessian) -> list[tuple[str, InputType]]:
        """Use to get the names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: Hessian) -> list[str]:
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
        return kornia.feature.hessian_response(image)


class HessianBlur(AbstractModel):
    """
    nn.Module wrapper for kornia.feature.Hessian_response(kornia.filters.gaussian_blur2d).

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
    def model_type(cls: HessianBlur) -> ModelType:
        """Use to get the type of input this model takes."""
        return ModelType.KERNEL

    @classmethod
    def input_names(cls: HessianBlur) -> list[tuple[str, InputType]]:
        """Use to get the names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: HessianBlur) -> list[str]:
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
        return kornia.feature.hessian_response(
            kornia.filters.gaussian_blur2d(
                image,
                (self._kernel_size, self._kernel_size),
                (self._sigma, self._sigma),
            )
        )


class HessianGray(AbstractModel):
    """
    nn.Module wrapper for kornia.feature.Hessian_response, with grayscale output.

    Methods
    -------
    forward(image: torch.Tensor) -> torch.Tensor
        Use to run the model on the input tensor.
    """

    def __init__(self: Self) -> None:
        """Use to create an instance of the model."""
        super().__init__()

    @classmethod
    def model_type(cls: HessianGray) -> ModelType:
        """Use to get the type of input this model takes."""
        return ModelType.NONE

    @classmethod
    def input_names(cls: HessianGray) -> list[tuple[str, InputType]]:
        """Use to get the names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: HessianGray) -> list[str]:
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
        hessian = kornia.feature.hessian_response(image)
        return kornia.color.bgr_to_grayscale(hessian)


class HessianBlurGray(AbstractModel):
    """
    nn.Module wrapper for kornia.feature.Hessian_response(kornia.filters.gaussian_blur2d), with grayscale output.

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
    def model_type(cls: HessianBlurGray) -> ModelType:
        """Use to get the type of input this model takes."""
        return ModelType.KERNEL

    @classmethod
    def input_names(cls: HessianBlurGray) -> list[tuple[str, InputType]]:
        """Use to get the names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: HessianBlurGray) -> list[str]:
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
        hessian = kornia.feature.hessian_response(
            kornia.filters.gaussian_blur2d(
                image,
                (self._kernel_size, self._kernel_size),
                (self._sigma, self._sigma),
            )
        )
        return kornia.color.bgr_to_grayscale(hessian)
