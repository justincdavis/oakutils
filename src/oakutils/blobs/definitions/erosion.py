from __future__ import annotations

from typing import TYPE_CHECKING

import kornia

from .abstract_model import AbstractModel
from .utils import InputType, ModelType

if TYPE_CHECKING:
    import torch
    from typing_extensions import Self


class Erosion(AbstractModel):
    """nn.Module wrapper for kornia.morphology.erosion."""

    def __init__(self: Self, kernel_size: int = 3) -> None:
        super().__init__()
        self._kernel_size = kernel_size

    @classmethod
    def model_type(cls: Erosion) -> ModelType:
        """The type of input this model takes."""
        return ModelType.KERNEL

    @classmethod
    def input_names(cls: Erosion) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: Erosion) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        return kornia.morphology.erosion(image, (self._kernel_size, self._kernel_size))


class ErosionGray(AbstractModel):
    """nn.Module wrapper for kornia.morphology.erosion, with grayscale output."""

    def __init__(self: Self, kernel_size: int = 3) -> None:
        super().__init__()
        self._kernel_size = kernel_size

    @classmethod
    def model_type(cls: ErosionGray) -> ModelType:
        """The type of input this model takes."""
        return ModelType.KERNEL

    @classmethod
    def input_names(cls: ErosionGray) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: ErosionGray) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        erosion = kornia.morphology.erosion(image, (self._kernel_size, self._kernel_size))
        return kornia.color.bgr_to_grayscale(erosion)


class ErosionBlur(AbstractModel):
    """nn.Module wrapper for kornia.morphology.erosion, with gaussian blur."""

    def __init__(
        self: Self, kernel_size: int = 3, kernel_size2: int = 3, sigma: float = 1.5
    ) -> None:
        super().__init__()
        self._kernel_size = kernel_size
        self._kernel_size2 = kernel_size2
        self._sigma = sigma

    @classmethod
    def model_type(cls: ErosionBlur) -> ModelType:
        """The type of input this model takes."""
        return ModelType.DUAL_KERNEL

    @classmethod
    def input_names(cls: ErosionBlur) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: ErosionBlur) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        gaussian = kornia.filters.gaussian_blur2d(
            image, (self._kernel_size, self._kernel_size), (self._sigma, self._sigma)
        )
        return kornia.morphology.erosion(gaussian, (self._kernel_size2, self._kernel_size2))


class ErosionBlurGray(AbstractModel):
    """nn.Module wrapper for kornia.morphology.erosion,
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
    def model_type(cls: ErosionBlurGray) -> ModelType:
        """The type of input this model takes."""
        return ModelType.DUAL_KERNEL

    @classmethod
    def input_names(cls: ErosionBlurGray) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: ErosionBlurGray) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        gaussian = kornia.filters.gaussian_blur2d(
            image, (self._kernel_size, self._kernel_size), (self._sigma, self._sigma)
        )
        erosion = kornia.morphology.erosion(gaussian, (self._kernel_size2, self._kernel_size2))
        return kornia.color.bgr_to_grayscale(erosion)
