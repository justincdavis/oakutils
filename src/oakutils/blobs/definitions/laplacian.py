from typing import List

import kornia
import torch

from .abstract_model import AbstractModel, ModelInput


class Laplacian(AbstractModel):
    """
    nn.Module wrapper for kornia.filters.laplacian
    """

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self._kernel_size = kernel_size

    @classmethod
    def input_type(self) -> ModelInput:
        """
        The type of input this model takes
        """
        return ModelInput.COLOR

    @classmethod
    def input_names(self) -> List[str]:
        """
        The names of the input tensors
        """
        return ["input"]

    @classmethod
    def output_names(self) -> List[str]:
        """
        The names of the output tensors
        """
        return ["output"]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return kornia.filters.laplacian(image, self._kernel_size)


class LaplacianGray(AbstractModel):
    """
    nn.Module wrapper for kornia.filters.laplacian, with grayscale output
    """

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self._kernel_size = kernel_size

    @classmethod
    def input_type(self) -> ModelInput:
        """
        The type of input this model takes
        """
        return ModelInput.COLOR

    @classmethod
    def input_names(self) -> List[str]:
        """
        The names of the input tensors
        """
        return ["input"]

    @classmethod
    def output_names(self) -> List[str]:
        """
        The names of the output tensors
        """
        return ["output"]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        laplacian = kornia.filters.laplacian(image, self._kernel_size)
        normalized = kornia.enhance.normalize_min_max(laplacian)
        return kornia.color.bgr_to_grayscale(normalized)


class LaplacianBlur(AbstractModel):
    """
    nn.Module wrapper for kornia.filters.laplacian, with gaussian blur
    """

    def __init__(self, kernel_size: int = 3, sigma: float = 0.5):
        super().__init__()
        self._kernel_size = kernel_size
        self._sigma = sigma

    @classmethod
    def input_type(self) -> ModelInput:
        """
        The type of input this model takes
        """
        return ModelInput.COLOR

    @classmethod
    def input_names(self) -> List[str]:
        """
        The names of the input tensors
        """
        return ["input"]

    @classmethod
    def output_names(self) -> List[str]:
        """
        The names of the output tensors
        """
        return ["output"]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        gaussian = kornia.filters.gaussian_blur2d(
            image, (self._kernel_size, self._kernel_size), (self._sigma, self._sigma)
        )
        return kornia.filters.laplacian(gaussian, self._kernel_size)


class LaplacianBlurGray(AbstractModel):
    """
    nn.Module wrapper for kornia.filters.laplacian, with gaussian blur, that outputs grayscale
    """

    def __init__(self, kernel_size: int = 3, sigma: float = 0.5):
        super().__init__()
        self._kernel_size = kernel_size
        self._sigma = sigma

    @classmethod
    def input_type(self) -> ModelInput:
        """
        The type of input this model takes
        """
        return ModelInput.COLOR

    @classmethod
    def input_names(self) -> List[str]:
        """
        The names of the input tensors
        """
        return ["input"]

    @classmethod
    def output_names(self) -> List[str]:
        """
        The names of the output tensors
        """
        return ["output"]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        gaussian = kornia.filters.gaussian_blur2d(
            image, (self._kernel_size, self._kernel_size), (self._sigma, self._sigma)
        )
        laplacian = kornia.filters.laplacian(gaussian, self._kernel_size)
        normalized = kornia.enhance.normalize_min_max(laplacian)
        return kornia.color.bgr_to_grayscale(normalized)
