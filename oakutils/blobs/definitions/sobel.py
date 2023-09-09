from __future__ import annotations

import kornia
import torch

from .abstract_model import AbstractModel, InputType, ModelType


class Sobel(AbstractModel):
    """nn.Module wrapper for kornia.filters.sobel."""

    def __init__(self):
        super().__init__()

    @classmethod
    def model_type(cls) -> ModelType:
        """The type of input this model takes."""
        return ModelType.NONE

    @classmethod
    def input_names(cls) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return kornia.filters.sobel(image)


class SobelBlur(AbstractModel):
    """nn.Module wrapper for kornia.filters.sobel(kornia.filters.gaussian_blur2d)."""

    def __init__(self, kernel_size: int = 3, sigma: float = 0.5):
        super().__init__()
        self._kernel_size = kernel_size
        self._sigma = sigma

    @classmethod
    def model_type(cls) -> ModelType:
        """The type of input this model takes."""
        return ModelType.KERNEL

    @classmethod
    def input_names(cls) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return kornia.filters.sobel(
            kornia.filters.gaussian_blur2d(
                image,
                (self._kernel_size, self._kernel_size),
                (self._sigma, self._sigma),
            )
        )


class SobelGray(AbstractModel):
    """nn.Module wrapper for kornia.filters.sobel, with grayscale output."""

    def __init__(self):
        super().__init__()

    @classmethod
    def model_type(cls) -> ModelType:
        """The type of input this model takes."""
        return ModelType.NONE

    @classmethod
    def input_names(cls) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        sobel = kornia.filters.sobel(image)
        normalized = kornia.enhance.normalize_min_max(sobel)
        return kornia.color.bgr_to_grayscale(normalized)


class SobelBlurGray(AbstractModel):
    """nn.Module wrapper for
    kornia.filters.sobel(kornia.filters.gaussian_blur2d),
      with grayscale output.
    """

    def __init__(self, kernel_size: int = 3, sigma: float = 0.5):
        super().__init__()
        self._kernel_size = kernel_size
        self._sigma = sigma

    @classmethod
    def model_type(cls) -> ModelType:
        """The type of input this model takes."""
        return ModelType.KERNEL

    @classmethod
    def input_names(cls) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        sobel = kornia.filters.sobel(
            kornia.filters.gaussian_blur2d(
                image,
                (self._kernel_size, self._kernel_size),
                (self._sigma, self._sigma),
            )
        )
        normalized = kornia.enhance.normalize_min_max(sobel)
        return kornia.color.bgr_to_grayscale(normalized)
