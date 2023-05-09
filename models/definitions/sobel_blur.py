import kornia
from torch import nn


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
