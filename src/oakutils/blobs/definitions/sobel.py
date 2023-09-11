from __future__ import annotations

from typing import TYPE_CHECKING

import kornia

from .abstract_model import AbstractModel
from .utils import InputType, ModelType

if TYPE_CHECKING:
    import torch
    from typing_extensions import Self


class Sobel(AbstractModel):
    """nn.Module wrapper for kornia.filters.sobel."""

    def __init__(self: Self) -> None:
        super().__init__()

    @classmethod
    def model_type(cls: Sobel) -> ModelType:
        """The type of input this model takes."""
        return ModelType.NONE

    @classmethod
    def input_names(cls: Sobel) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: Sobel) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        return kornia.filters.sobel(image)


class SobelBlur(AbstractModel):
    """nn.Module wrapper for kornia.filters.sobel(kornia.filters.gaussian_blur2d)."""

    def __init__(self: Self, kernel_size: int = 3, sigma: float = 0.5) -> None:
        super().__init__()
        self._kernel_size = kernel_size
        self._sigma = sigma

    @classmethod
    def model_type(cls: SobelBlur) -> ModelType:
        """The type of input this model takes."""
        return ModelType.KERNEL

    @classmethod
    def input_names(cls: SobelBlur) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: SobelBlur) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        return kornia.filters.sobel(
            kornia.filters.gaussian_blur2d(
                image,
                (self._kernel_size, self._kernel_size),
                (self._sigma, self._sigma),
            )
        )


class SobelGray(AbstractModel):
    """nn.Module wrapper for kornia.filters.sobel, with grayscale output."""

    def __init__(self: Self) -> None:
        super().__init__()

    @classmethod
    def model_type(cls: SobelGray) -> ModelType:
        """The type of input this model takes."""
        return ModelType.NONE

    @classmethod
    def input_names(cls: SobelGray) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: SobelGray) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        sobel = kornia.filters.sobel(image)
        normalized = kornia.enhance.normalize_min_max(sobel)
        return kornia.color.bgr_to_grayscale(normalized)


class SobelBlurGray(AbstractModel):
    """nn.Module wrapper for
    kornia.filters.sobel(kornia.filters.gaussian_blur2d),
      with grayscale output.
    """

    def __init__(self: Self, kernel_size: int = 3, sigma: float = 0.5) -> None:
        super().__init__()
        self._kernel_size = kernel_size
        self._sigma = sigma

    @classmethod
    def model_type(cls: SobelBlurGray) -> ModelType:
        """The type of input this model takes."""
        return ModelType.KERNEL

    @classmethod
    def input_names(cls: SobelBlurGray) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: SobelBlurGray) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        sobel = kornia.filters.sobel(
            kornia.filters.gaussian_blur2d(
                image,
                (self._kernel_size, self._kernel_size),
                (self._sigma, self._sigma),
            )
        )
        normalized = kornia.enhance.normalize_min_max(sobel)
        return kornia.color.bgr_to_grayscale(normalized)
