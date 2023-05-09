import kornia
from torch import nn


class Gaussian(nn.Module):
    """
    nn.Module wrapper for kornia.filters.gaussian_blur2d
    """
    
    def __init__(self, kernel_size=3, sigma=1.5):
        super().__init__()
        self._kernel_size = kernel_size
        self._sigma = sigma
    
    def forward(self, image):
        return kornia.filters.gaussian_blur2d(image, (self._kernel_size, self._kernel_size), (self._sigma, self._sigma))
