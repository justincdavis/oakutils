from __future__ import annotations

from typing import TYPE_CHECKING

import kornia

from .abstract_model import AbstractModel
from .utils import InputType, ModelType

if TYPE_CHECKING:
    import torch
    from typing_extensions import Self


class Opening(AbstractModel):
    """nn.Module wrapper for kornia.morphology.opening."""

    def __init__(self: Self, kernel_size: int = 3) -> None:
        super().__init__()
        self._kernel_size = kernel_size

    @classmethod
    def model_type(cls: Opening) -> ModelType:
        """The type of input this model takes."""
        return ModelType.KERNEL

    @classmethod
    def input_names(cls: Opening) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: Opening) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        return kornia.morphology.opening(image, (self._kernel_size, self._kernel_size))


class OpeningGray(AbstractModel):
    """nn.Module wrapper for kornia.morphology.opening, with grayscale output."""

    def __init__(self: Self, kernel_size: int = 3) -> None:
        super().__init__()
        self._kernel_size = kernel_size

    @classmethod
    def model_type(cls: OpeningGray) -> ModelType:
        """The type of input this model takes."""
        return ModelType.KERNEL

    @classmethod
    def input_names(cls: OpeningGray) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: OpeningGray) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        opening = kornia.morphology.opening(image, (self._kernel_size, self._kernel_size))
        return kornia.color.bgr_to_grayscale(opening)


class OpeningBlur(AbstractModel):
    """nn.Module wrapper for kornia.morphology.opening, with gaussian blur."""

    def __init__(
        self: Self, kernel_size: int = 3, kernel_size2: int = 3, sigma: float = 1.5
    ) -> None:
        super().__init__()
        self._kernel_size = kernel_size
        self._kernel_size2 = kernel_size2
        self._sigma = sigma

    @classmethod
    def model_type(cls: OpeningBlur) -> ModelType:
        """The type of input this model takes."""
        return ModelType.DUAL_KERNEL

    @classmethod
    def input_names(cls: OpeningBlur) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: OpeningBlur) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        gaussian = kornia.filters.gaussian_blur2d(
            image, (self._kernel_size, self._kernel_size), (self._sigma, self._sigma)
        )
        return kornia.morphology.opening(gaussian, (self._kernel_size2, self._kernel_size2))


class OpeningBlurGray(AbstractModel):
    """nn.Module wrapper for kornia.morphology.opening,
    with gaussian blur, that outputs grayscale.
    """

    def __init__(
        self: Self, kernel_size: int = 3, kernel_size2: int = 3, sigma: float = 1.5
    ) -> None:
        super().__init__()
        self._kernel_size = kernel_size
        self._kernel_size2 = kernel_size2
        self._sigma = sigma

    @classmethod
    def model_type(cls: OpeningBlurGray) -> ModelType:
        """The type of input this model takes."""
        return ModelType.DUAL_KERNEL

    @classmethod
    def input_names(cls: OpeningBlurGray) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: OpeningBlurGray) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        gaussian = kornia.filters.gaussian_blur2d(
            image, (self._kernel_size, self._kernel_size), (self._sigma, self._sigma)
        )
        opening = kornia.morphology.opening(gaussian, (self._kernel_size2, self._kernel_size2))
        return kornia.color.bgr_to_grayscale(opening)
