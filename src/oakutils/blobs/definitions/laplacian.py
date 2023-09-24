from __future__ import annotations

from typing import TYPE_CHECKING

import kornia

from .abstract_model import AbstractModel
from .utils import InputType, ModelType

if TYPE_CHECKING:
    import torch
    from typing_extensions import Self


class Laplacian(AbstractModel):
    """nn.Module wrapper for kornia.filters.laplacian."""

    def __init__(self: Self, kernel_size: int = 3) -> None:
        super().__init__()
        self._kernel_size = kernel_size

    @classmethod
    def model_type(cls: Laplacian) -> ModelType:
        """The type of input this model takes."""
        return ModelType.KERNEL

    @classmethod
    def input_names(cls: Laplacian) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: Laplacian) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        return kornia.filters.laplacian(image, self._kernel_size)


class LaplacianGray(AbstractModel):
    """nn.Module wrapper for kornia.filters.laplacian, with grayscale output."""

    def __init__(self: Self, kernel_size: int = 3) -> None:
        super().__init__()
        self._kernel_size = kernel_size

    @classmethod
    def model_type(cls: LaplacianGray) -> ModelType:
        """The type of input this model takes."""
        return ModelType.KERNEL

    @classmethod
    def input_names(cls: LaplacianGray) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: LaplacianGray) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        laplacian = kornia.filters.laplacian(image, self._kernel_size)
        normalized = kornia.enhance.normalize_min_max(laplacian)
        return kornia.color.bgr_to_grayscale(normalized)


class LaplacianBlur(AbstractModel):
    """nn.Module wrapper for kornia.filters.laplacian, with gaussian blur."""

    def __init__(
        self: Self, kernel_size: int = 3, kernel_size2: int = 3, sigma: float = 0.5
    ) -> None:
        super().__init__()
        self._kernel_size = kernel_size
        self._kernel_size2 = kernel_size2
        self._sigma = sigma

    @classmethod
    def model_type(cls: LaplacianBlur) -> ModelType:
        """The type of input this model takes."""
        return ModelType.DUAL_KERNEL

    @classmethod
    def input_names(cls: LaplacianBlur) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: LaplacianBlur) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        gaussian = kornia.filters.gaussian_blur2d(
            image, (self._kernel_size, self._kernel_size), (self._sigma, self._sigma)
        )
        return kornia.filters.laplacian(gaussian, self._kernel_size2)


class LaplacianBlurGray(AbstractModel):
    """nn.Module wrapper for kornia.filters.laplacian,
    with gaussian blur, that outputs grayscale.
    """

    def __init__(
        self: Self, kernel_size: int = 3, kernel_size2: int = 3, sigma: float = 0.5
    ) -> None:
        super().__init__()
        self._kernel_size = kernel_size
        self._kernel_size2 = kernel_size2
        self._sigma = sigma

    @classmethod
    def model_type(cls: LaplacianBlurGray) -> ModelType:
        """The type of input this model takes."""
        return ModelType.DUAL_KERNEL

    @classmethod
    def input_names(cls: LaplacianBlurGray) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: LaplacianBlurGray) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        gaussian = kornia.filters.gaussian_blur2d(
            image, (self._kernel_size, self._kernel_size), (self._sigma, self._sigma)
        )
        laplacian = kornia.filters.laplacian(gaussian, self._kernel_size2)
        normalized = kornia.enhance.normalize_min_max(laplacian)
        return kornia.color.bgr_to_grayscale(normalized)
