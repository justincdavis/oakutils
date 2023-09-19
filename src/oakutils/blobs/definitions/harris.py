from __future__ import annotations

from typing import TYPE_CHECKING

import kornia

from .abstract_model import AbstractModel
from .utils import InputType, ModelType

if TYPE_CHECKING:
    import torch
    from typing_extensions import Self


class Harris(AbstractModel):
    """nn.Module wrapper for kornia.feature.harris."""

    def __init__(self: Self) -> None:
        super().__init__()

    @classmethod
    def model_type(cls: Harris) -> ModelType:
        """The type of input this model takes."""
        return ModelType.NONE

    @classmethod
    def input_names(cls: Harris) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: Harris) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        return kornia.feature.harris_response(image)


class HarrisBlur(AbstractModel):
    """nn.Module wrapper for kornia.feature.harris_response(kornia.filters.gaussian_blur2d)."""

    def __init__(self: Self, kernel_size: int = 3, sigma: float = 1.5) -> None:
        super().__init__()
        self._kernel_size = kernel_size
        self._sigma = sigma

    @classmethod
    def model_type(cls: HarrisBlur) -> ModelType:
        """The type of input this model takes."""
        return ModelType.KERNEL

    @classmethod
    def input_names(cls: HarrisBlur) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: HarrisBlur) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        return kornia.feature.harris_response(
            kornia.filters.gaussian_blur2d(
                image,
                (self._kernel_size, self._kernel_size),
                (self._sigma, self._sigma),
            )
        )


class HarrisGray(AbstractModel):
    """nn.Module wrapper for kornia.feature.harris_response, with grayscale output."""

    def __init__(self: Self) -> None:
        super().__init__()

    @classmethod
    def model_type(cls: HarrisGray) -> ModelType:
        """The type of input this model takes."""
        return ModelType.NONE

    @classmethod
    def input_names(cls: HarrisGray) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: HarrisGray) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        harris = kornia.feature.harris_response(image)
        return kornia.color.bgr_to_grayscale(harris)


class HarrisBlurGray(AbstractModel):
    """nn.Module wrapper for
    kornia.feature.harris_response(kornia.filters.gaussian_blur2d),
      with grayscale output.
    """

    def __init__(self: Self, kernel_size: int = 3, sigma: float = 1.5) -> None:
        super().__init__()
        self._kernel_size = kernel_size
        self._sigma = sigma

    @classmethod
    def model_type(cls: HarrisBlurGray) -> ModelType:
        """The type of input this model takes."""
        return ModelType.KERNEL

    @classmethod
    def input_names(cls: HarrisBlurGray) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: HarrisBlurGray) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        harris = kornia.feature.harris_response(
            kornia.filters.gaussian_blur2d(
                image,
                (self._kernel_size, self._kernel_size),
                (self._sigma, self._sigma),
            )
        )
        return kornia.color.bgr_to_grayscale(harris)
