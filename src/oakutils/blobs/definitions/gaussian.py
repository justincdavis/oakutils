from __future__ import annotations

from typing import TYPE_CHECKING

import kornia

from .abstract_model import AbstractModel
from .utils import InputType, ModelType

if TYPE_CHECKING:
    import torch
    from typing_extensions import Self


class Gaussian(AbstractModel):
    """nn.Module wrapper for kornia.filters.gaussian_blur2d."""

    def __init__(self: Self, kernel_size: int = 3, sigma: float = 0.5) -> None:
        super().__init__()
        self._kernel_size = kernel_size
        self._sigma = sigma

    @classmethod
    def model_type(cls: Gaussian) -> ModelType:
        """The type of input this model takes."""
        return ModelType.KERNEL

    @classmethod
    def input_names(cls: Gaussian) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: Gaussian) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        return kornia.filters.gaussian_blur2d(
            image, (self._kernel_size, self._kernel_size), (self._sigma, self._sigma)
        )


class GaussianGray(AbstractModel):
    """nn.Module wrapper for kornia.filters.gaussian_blur2d, with grayscale output."""

    def __init__(self: Self, kernel_size: int = 3, sigma: float = 0.5) -> None:
        super().__init__()
        self._kernel_size = kernel_size
        self._sigma = sigma

    @classmethod
    def model_type(cls: GaussianGray) -> ModelType:
        """The type of input this model takes."""
        return ModelType.KERNEL

    @classmethod
    def input_names(cls: GaussianGray) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: GaussianGray) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        gaussian = kornia.filters.gaussian_blur2d(
            image, (self._kernel_size, self._kernel_size), (self._sigma, self._sigma)
        )
        normalized = kornia.enhance.normalize_min_max(gaussian)
        return kornia.color.bgr_to_grayscale(normalized)
