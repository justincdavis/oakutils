import kornia
import torch


class DepthFilter(torch.nn.Module):
    """
    Filters the depth stream to isolate change in depth

    Performs the following steps:
    1. Gaussian blur of kernel_size
    2. Laplacian of kernel_size
    3. Normalization to 0-1
    4. Depth bias subtraction
    5. Normalization to 0-1
    """
    def __init__(self, kernel_size: int = 3, sigma: float = 0.5):
        super().__init__()
        self._kernel_size = kernel_size
        self._sigma = sigma

    def forward(self, image):
        max = torch.max(image)
        min = torch.min(image)
        gaussian = kornia.filters.gaussian_blur2d(
            image, (self._kernel_size, self._kernel_size), (self._sigma, self._sigma)
        )
        laplacian = kornia.filters.laplacian(gaussian, self._kernel_size)
        laplacian = kornia.enhance.normalize_min_max(laplacian, min_val=0.0, max_val=1.0)
        scale = 1.0 - ((gaussian - min) / (max - min))
        laplacian = laplacian + scale
        laplacian = kornia.enhance.normalize_min_max(laplacian, min_val=0.0, max_val=1.0)
        return laplacian
