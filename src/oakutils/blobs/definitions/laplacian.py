import kornia
from torch import nn


class Laplacian(nn.Module):
    """
    nn.Module wrapper for kornia.filters.laplacian
    """

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self._kernel_size = kernel_size

    def forward(self, image):
        return kornia.filters.laplacian(image, self._kernel_size)

class LaplacianGray(nn.Module):
    """
    nn.Module wrapper for kornia.filters.laplacian, with grayscale output
    """

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self._kernel_size = kernel_size

    def forward(self, image):
        laplacian = kornia.filters.laplacian(image, self._kernel_size)
        normalized = kornia.enhance.normalize_min_max(laplacian)
        return kornia.color.bgr_to_grayscale(normalized)

class LaplacianBlur(nn.Module):
    """
    nn.Module wrapper for kornia.filters.laplacian, with gaussian blur
    """

    def __init__(self, kernel_size: int = 3, sigma: float = 0.5):
        super().__init__()
        self._kernel_size = kernel_size
        self._sigma = sigma

    def forward(self, image):
        gaussian = kornia.filters.gaussian_blur2d(
            image, (self._kernel_size, self._kernel_size), (self._sigma, self._sigma)
        )
        return kornia.filters.laplacian(gaussian, self._kernel_size)
    
class LaplacianBlurGray(nn.Module):
    """
    nn.Module wrapper for kornia.filters.laplacian, with gaussian blur, that outputs grayscale
    """

    def __init__(self, kernel_size: int = 3, sigma: float = 0.5):
        super().__init__()
        self._kernel_size = kernel_size
        self._sigma = sigma

    def forward(self, image):
        gaussian = kornia.filters.gaussian_blur2d(
            image, (self._kernel_size, self._kernel_size), (self._sigma, self._sigma)
        )
        laplacian = kornia.filters.laplacian(gaussian, self._kernel_size)
        normalized = kornia.enhance.normalize_min_max(laplacian)
        return kornia.color.bgr_to_grayscale(normalized)
