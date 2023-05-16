import kornia
from torch import nn


class GaussianGray(nn.Module):
    """
    nn.Module wrapper for kornia.filters.gaussian_blur2d, with grayscale output
    """

    def __init__(self, kernel_size: int = 3, sigma: float = 0.5):
        super().__init__()
        self._kernel_size = kernel_size
        self._sigma = sigma

    def forward(self, image):
        gaussian = kornia.filters.gaussian_blur2d(
            image, (self._kernel_size, self._kernel_size), (self._sigma, self._sigma)
        )
        normalized = kornia.enhance.normalize_min_max(gaussian)
        return kornia.color.bgr_to_grayscale(normalized)
