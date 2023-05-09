import kornia
from torch import nn


class Laplacian(nn.Module):
    """
    nn.Module wrapper for kornia.filters.laplacian
    """
    
    def __init__(self, kernel_size=3):
        super().__init__()
        self._kernel_size = kernel_size

    def forward(self, image):
        return kornia.filters.laplacian(image, self._kernel_size)
