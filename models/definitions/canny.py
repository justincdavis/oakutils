import kornia
from torch import nn


class Canny(nn.Module):
    """
    nn.Module wrapper for kornia.filters.canny
    """

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self._kernel_size = kernel_size

    def forward(self, image):
        return kornia.filters.canny(
            image, kernel_size=(self._kernel_size, self._kernel_size)
        )
