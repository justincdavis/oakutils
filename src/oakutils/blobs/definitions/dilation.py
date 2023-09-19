from __future__ import annotations

from typing import TYPE_CHECKING

import kornia

from .abstract_model import AbstractModel
from .utils import InputType, ModelType

if TYPE_CHECKING:
    import torch
    from typing_extensions import Self


class Dilation(AbstractModel):
    """nn.Module wrapper for kornia.morphology.dilation."""

    def __init__(self: Self, kernel_size: int = 3) -> None:
        super().__init__()
        self._kernel_size = kernel_size

    @classmethod
    def model_type(cls: Dilation) -> ModelType:
        """The type of input this model takes."""
        return ModelType.KERNEL

    @classmethod
    def input_names(cls: Dilation) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: Dilation) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        return kornia.morphology.dilation(image, (self._kernel_size, self._kernel_size))


class DilationGray(AbstractModel):
    """nn.Module wrapper for kornia.morphology.dilation, with grayscale output."""

    def __init__(self: Self, kernel_size: int = 3) -> None:
        super().__init__()
        self._kernel_size = kernel_size

    @classmethod
    def model_type(cls: DilationGray) -> ModelType:
        """The type of input this model takes."""
        return ModelType.KERNEL

    @classmethod
    def input_names(cls: DilationGray) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: DilationGray) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        dilation = kornia.morphology.dilation(image, (self._kernel_size, self._kernel_size))
        return kornia.color.bgr_to_grayscale(dilation)


class DilationBlur(AbstractModel):
    """nn.Module wrapper for kornia.morphology.dilation, with gaussian blur."""

    def __init__(
        self: Self, kernel_size: int = 3, kernel_size2: int = 3, sigma: float = 1.5
    ) -> None:
        super().__init__()
        self._kernel_size = kernel_size
        self._kernel_size2 = kernel_size2
        self._sigma = sigma

    @classmethod
    def model_type(cls: DilationBlur) -> ModelType:
        """The type of input this model takes."""
        return ModelType.DUAL_KERNEL

    @classmethod
    def input_names(cls: DilationBlur) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: DilationBlur) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        gaussian = kornia.filters.gaussian_blur2d(
            image, (self._kernel_size, self._kernel_size), (self._sigma, self._sigma)
        )
        return kornia.morphology.dilation(gaussian, (self._kernel_size2, self._kernel_size2))


class DilationBlurGray(AbstractModel):
    """nn.Module wrapper for kornia.morphology.dilation,
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
    def model_type(cls: DilationBlurGray) -> ModelType:
        """The type of input this model takes."""
        return ModelType.DUAL_KERNEL

    @classmethod
    def input_names(cls: DilationBlurGray) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: DilationBlurGray) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        gaussian = kornia.filters.gaussian_blur2d(
            image, (self._kernel_size, self._kernel_size), (self._sigma, self._sigma)
        )
        dilation = kornia.morphology.dilation(gaussian, (self._kernel_size2, self._kernel_size2))
        return kornia.color.bgr_to_grayscale(dilation)
