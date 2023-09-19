from __future__ import annotations

from typing import TYPE_CHECKING

import kornia

from .abstract_model import AbstractModel
from .utils import InputType, ModelType

if TYPE_CHECKING:
    import torch
    from typing_extensions import Self


class Hessian(AbstractModel):
    """nn.Module wrapper for kornia.feature.Hessian."""

    def __init__(self: Self) -> None:
        super().__init__()

    @classmethod
    def model_type(cls: Hessian) -> ModelType:
        """The type of input this model takes."""
        return ModelType.NONE

    @classmethod
    def input_names(cls: Hessian) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: Hessian) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        return kornia.feature.hessian_response(image)


class HessianBlur(AbstractModel):
    """nn.Module wrapper for kornia.feature.Hessian_response(kornia.filters.gaussian_blur2d)."""

    def __init__(self: Self, kernel_size: int = 3, sigma: float = 1.5) -> None:
        super().__init__()
        self._kernel_size = kernel_size
        self._sigma = sigma

    @classmethod
    def model_type(cls: HessianBlur) -> ModelType:
        """The type of input this model takes."""
        return ModelType.KERNEL

    @classmethod
    def input_names(cls: HessianBlur) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: HessianBlur) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        return kornia.feature.hessian_response(
            kornia.filters.gaussian_blur2d(
                image,
                (self._kernel_size, self._kernel_size),
                (self._sigma, self._sigma),
            )
        )


class HessianGray(AbstractModel):
    """nn.Module wrapper for kornia.feature.Hessian_response, with grayscale output."""

    def __init__(self: Self) -> None:
        super().__init__()

    @classmethod
    def model_type(cls: HessianGray) -> ModelType:
        """The type of input this model takes."""
        return ModelType.NONE

    @classmethod
    def input_names(cls: HessianGray) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: HessianGray) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        hessian = kornia.feature.hessian_response(image)
        return kornia.color.bgr_to_grayscale(hessian)


class HessianBlurGray(AbstractModel):
    """nn.Module wrapper for
    kornia.feature.Hessian_response(kornia.filters.gaussian_blur2d),
      with grayscale output.
    """

    def __init__(self: Self, kernel_size: int = 3, sigma: float = 1.5) -> None:
        super().__init__()
        self._kernel_size = kernel_size
        self._sigma = sigma

    @classmethod
    def model_type(cls: HessianBlurGray) -> ModelType:
        """The type of input this model takes."""
        return ModelType.KERNEL

    @classmethod
    def input_names(cls: HessianBlurGray) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: HessianBlurGray) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        hessian = kornia.feature.hessian_response(
            kornia.filters.gaussian_blur2d(
                image,
                (self._kernel_size, self._kernel_size),
                (self._sigma, self._sigma),
            )
        )
        return kornia.color.bgr_to_grayscale(hessian)
