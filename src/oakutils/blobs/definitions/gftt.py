from __future__ import annotations

from typing import TYPE_CHECKING

import kornia

from .abstract_model import AbstractModel
from .utils import InputType, ModelType

if TYPE_CHECKING:
    import torch
    from typing_extensions import Self


class GFTT(AbstractModel):
    """nn.Module wrapper for kornia.feature.GFTT."""

    def __init__(self: Self) -> None:
        super().__init__()

    @classmethod
    def model_type(cls: GFTT) -> ModelType:
        """The type of input this model takes."""
        return ModelType.NONE

    @classmethod
    def input_names(cls: GFTT) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: GFTT) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        return kornia.feature.gftt_response(image)


class GFTTBlur(AbstractModel):
    """nn.Module wrapper for kornia.feature.GFTT_response(kornia.filters.gaussian_blur2d)."""

    def __init__(self: Self, kernel_size: int = 3, sigma: float = 1.5) -> None:
        super().__init__()
        self._kernel_size = kernel_size
        self._sigma = sigma

    @classmethod
    def model_type(cls: GFTTBlur) -> ModelType:
        """The type of input this model takes."""
        return ModelType.KERNEL

    @classmethod
    def input_names(cls: GFTTBlur) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: GFTTBlur) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        return kornia.feature.gftt_response(
            kornia.filters.gaussian_blur2d(
                image,
                (self._kernel_size, self._kernel_size),
                (self._sigma, self._sigma),
            )
        )


class GFTTGray(AbstractModel):
    """nn.Module wrapper for kornia.feature.GFTT_response, with grayscale output."""

    def __init__(self: Self) -> None:
        super().__init__()

    @classmethod
    def model_type(cls: GFTTGray) -> ModelType:
        """The type of input this model takes."""
        return ModelType.NONE

    @classmethod
    def input_names(cls: GFTTGray) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: GFTTGray) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        gftt = kornia.feature.gftt_response(image)
        return kornia.color.bgr_to_grayscale(gftt)


class GFTTBlurGray(AbstractModel):
    """nn.Module wrapper for
    kornia.feature.GFTT_response(kornia.filters.gaussian_blur2d),
      with grayscale output.
    """

    def __init__(self: Self, kernel_size: int = 3, sigma: float = 1.5) -> None:
        super().__init__()
        self._kernel_size = kernel_size
        self._sigma = sigma

    @classmethod
    def model_type(cls: GFTTBlurGray) -> ModelType:
        """The type of input this model takes."""
        return ModelType.KERNEL

    @classmethod
    def input_names(cls: GFTTBlurGray) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: GFTTBlurGray) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        gftt = kornia.feature.gftt_response(
            kornia.filters.gaussian_blur2d(
                image,
                (self._kernel_size, self._kernel_size),
                (self._sigma, self._sigma),
            )
        )
        return kornia.color.bgr_to_grayscale(gftt)
