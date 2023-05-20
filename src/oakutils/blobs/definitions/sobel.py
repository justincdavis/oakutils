import kornia
from torch import nn


class Sobel(nn.Module):
    """
    nn.Module wrapper for kornia.filters.sobel
    """

    def __init__(self):
        super().__init__()

    def forward(self, image):
        return kornia.filters.sobel(
            image
        )

class SobelBlur(nn.Module):
    """
    nn.Module wrapper for kornia.filters.sobel(kornia.filters.gaussian_blur2d)
    """

    def __init__(self, kernel_size: int = 3, sigma: float = 0.5):
        super().__init__()
        self._kernel_size = kernel_size
        self._sigma = sigma

    def forward(self, image):
        return kornia.filters.sobel(
            kornia.filters.gaussian_blur2d(
                image, (self._kernel_size, self._kernel_size), (self._sigma, self._sigma)
            )
        )

class SobelGray(nn.Module):
    """
    nn.Module wrapper for kornia.filters.sobel, with grayscale output
    """

    def __init__(self):
        super().__init__()

    def forward(self, image):
        sobel = kornia.filters.sobel(
            image
        )
        normalized = kornia.enhance.normalize_min_max(sobel)
        return kornia.color.bgr_to_grayscale(normalized)

class SobelBlurGray(nn.Module):
    """
    nn.Module wrapper for kornia.filters.sobel(kornia.filters.gaussian_blur2d), with grayscale output
    """

    def __init__(self, kernel_size: int = 3, sigma: float = 0.5):
        super().__init__()
        self._kernel_size = kernel_size
        self._sigma = sigma

    def forward(self, image):
        sobel = kornia.filters.sobel(
            kornia.filters.gaussian_blur2d(
                image, (self._kernel_size, self._kernel_size), (self._sigma, self._sigma)
            )
        )
        normalized = kornia.enhance.normalize_min_max(sobel)
        return kornia.color.bgr_to_grayscale(normalized)
