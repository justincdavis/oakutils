import torch

from definitions import Gaussian, Laplacian, Canny, Sobel, SobelBlur, LaplacianGray, GaussianGray, DepthFilter


COLOR__TO_COLOR_CLASSES = {
    "gaussian": Gaussian,
    "laplacian": Laplacian,
    "sobel": Sobel,
    "sobel_blur": SobelBlur,
}
COLOR_TO_GRAY_CLASSES = {
    "laplacian_gray": LaplacianGray,
    "gaussian_gray": GaussianGray,
}
COLOR_TO_TWO_GRAY_CLASSES = {
    "canny": Canny,
}
GRAY_TO_GRAY_CLASSES = {
    "depth_filter": DepthFilter,
    "grayscale_gaussian": Gaussian,
    "grayscale_laplacian": Laplacian,
    "grayscale_sobel": Sobel,
    "grayscale_sobel_blur": SobelBlur,
}
GRAY_TO_TWO_GRAY_CLASSES = {
    "grayscale_canny": Canny,
}

# size parameters
HEIGHT = 480
WIDTH = 640

# create some random data
gray = torch.rand(1, 1, HEIGHT, WIDTH)
color = torch.rand(1, 3, HEIGHT, WIDTH)

# run the models
result = {}
for name, model in COLOR__TO_COLOR_CLASSES.items():
    result[name] = model()(color)
    assert result[name].shape == (1, 3, HEIGHT, WIDTH)
for name, model in COLOR_TO_GRAY_CLASSES.items():
    result[name] = model()(color)
    assert result[name].shape == (1, 1, HEIGHT, WIDTH)
for name, model in COLOR_TO_TWO_GRAY_CLASSES.items():
    result[name] = model()(color)
    r1, r2 = result[name]
    assert r1.shape == (1, 1, HEIGHT, WIDTH)
    assert r2.shape == (1, 1, HEIGHT, WIDTH)
for name, model in GRAY_TO_GRAY_CLASSES.items():
    result[name] = model()(gray)
    assert result[name].shape == (1, 1, HEIGHT, WIDTH)
for name, model in GRAY_TO_TWO_GRAY_CLASSES.items():
    result[name] = model()(gray)
    r1, r2 = result[name]
    assert r1.shape == (1, 1, HEIGHT, WIDTH)
    assert r2.shape == (1, 1, HEIGHT, WIDTH)
