from __future__ import annotations

from typing import TYPE_CHECKING

import kornia

from .abstract_model import AbstractModel
from .utils import InputType, ModelType

if TYPE_CHECKING:
    import torch
    from typing_extensions import Self


class Closing(AbstractModel):
    """nn.Module wrapper for kornia.morphology.closing."""

    def __init__(self: Self, kernel_size: int = 3) -> None:
        super().__init__()
        self._kernel_size = kernel_size

    @classmethod
    def model_type(cls: Closing) -> ModelType:
        """The type of input this model takes."""
        return ModelType.KERNEL

    @classmethod
    def input_names(cls: Closing) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: Closing) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        return kornia.morphology.closing(image, (self._kernel_size, self._kernel_size))


class ClosingGray(AbstractModel):
    """nn.Module wrapper for kornia.morphology.closing, with grayscale output."""

    def __init__(self: Self, kernel_size: int = 3) -> None:
        super().__init__()
        self._kernel_size = kernel_size

    @classmethod
    def model_type(cls: ClosingGray) -> ModelType:
        """The type of input this model takes."""
        return ModelType.KERNEL

    @classmethod
    def input_names(cls: ClosingGray) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: ClosingGray) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        closing = kornia.morphology.closing(image, (self._kernel_size, self._kernel_size))
        return kornia.color.bgr_to_grayscale(closing)


class ClosingBlur(AbstractModel):
    """nn.Module wrapper for kornia.morphology.closing, with gaussian blur."""

    def __init__(
        self: Self, kernel_size: int = 3, kernel_size2: int = 3, sigma: float = 1.5
    ) -> None:
        super().__init__()
        self._kernel_size = kernel_size
        self._kernel_size2 = kernel_size2
        self._sigma = sigma

    @classmethod
    def model_type(cls: ClosingBlur) -> ModelType:
        """The type of input this model takes."""
        return ModelType.DUAL_KERNEL

    @classmethod
    def input_names(cls: ClosingBlur) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: ClosingBlur) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        gaussian = kornia.filters.gaussian_blur2d(
            image, (self._kernel_size, self._kernel_size), (self._sigma, self._sigma)
        )
        return kornia.morphology.closing(gaussian, (self._kernel_size2, self._kernel_size2))


class ClosingBlurGray(AbstractModel):
    """nn.Module wrapper for kornia.morphology.closing,
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
    def model_type(cls: ClosingBlurGray) -> ModelType:
        """The type of input this model takes."""
        return ModelType.DUAL_KERNEL

    @classmethod
    def input_names(cls: ClosingBlurGray) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: ClosingBlurGray) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        gaussian = kornia.filters.gaussian_blur2d(
            image, (self._kernel_size, self._kernel_size), (self._sigma, self._sigma)
        )
        closing = kornia.morphology.closing(gaussian, (self._kernel_size2, self._kernel_size2))
        return kornia.color.bgr_to_grayscale(closing)
