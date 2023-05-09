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
