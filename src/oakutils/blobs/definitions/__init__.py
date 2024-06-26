# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Module defining definitions for blobs.

Classes
-------
AbstractModel
    Abstract base class for models.
Closing
    nn.Module wrapper for kornia.morphology.closing.
ClosingBlur
    nn.Module wrapper for kornia.morphology.closing, with gaussian blur.
ClosingGray
    nn.Module wrapper for kornia.morphology.closing, with grayscale output.
ClosingBlurGray
    nn.Module wrapper for kornia.morphology.closing, with grayscale output and gaussian blur.
Dilation
    nn.Module wrapper for kornia.morphology.dilation.
DilationBlur
    nn.Module wrapper for kornia.morphology.dilation, with gaussian blur.
DilationGray
    nn.Module wrapper for kornia.morphology.dilation, with grayscale output.
DilationBlurGray
    nn.Module wrapper for kornia.morphology.dilation, with grayscale output and gaussian blur.
Erosion
    nn.Module wrapper for kornia.morphology.erosion.
ErosionBlur
    nn.Module wrapper for kornia.morphology.erosion, with gaussian blur.
ErosionGray
    nn.Module wrapper for kornia.morphology.erosion, with grayscale output.
ErosionBlurGray
    nn.Module wrapper for kornia.morphology.erosion, with grayscale output and gaussian blur.
Gaussian
    nn.Module wrapper for kornia.filters.gaussian_blur2d.
GaussianGray
    nn.Module wrapper for kornia.filters.gaussian_blur2d, with grayscale output.
GFTT
    nn.Module wrapper for kornia.filters.gftt_response.
GFTTBlur
    nn.Module wrapper for kornia.filters.gftt_response, with gaussian blur.
GFTTGray
    nn.Module wrapper for kornia.filters.gftt_response, with grayscale output.
GFTTBlurGray
    nn.Module wrapper for kornia.filters.gftt_response, with grayscale output and gaussian blur.
Harris
    nn.Module wrapper for kornia.feature.harris_response.
HarrisBlur
    nn.Module wrapper for kornia.feature.harris_response, with gaussian blur.
HarrisGray
    nn.Module wrapper for kornia.feature.harris_response, with grayscale output.
HarrisBlurGray
    nn.Module wrapper for kornia.feature.harris_response, with grayscale output and gaussian blur.
Hessian
    nn.Module wrapper for kornia.feature.Hessian_response.
HessianBlur
    nn.Module wrapper for kornia.feature.Hessian_response, with gaussian blur.
HessianGray
    nn.Module wrapper for kornia.feature.Hessian_response, with grayscale output.
HessianBlurGray
    nn.Module wrapper for kornia.feature.Hessian_response, with grayscale output and gaussian blur.
Laserscan
    nn.Module for creating a laserscan from a depth image.
Laplacian
    nn.Module wrapper for kornia.filters.laplacian.
LaplacianBlur
    nn.Module wrapper for kornia.filters.laplacian, with gaussian blur.
LaplacianGray
    nn.Module wrapper for kornia.filters.laplacian, with grayscale output.
LaplacianBlurGray
    nn.Module wrapper for kornia.filters.laplacian, with grayscale output and gaussian blur.
Opening
    nn.Module wrapper for kornia.morphology.opening.
OpeningBlur
    nn.Module wrapper for kornia.morphology.opening, with gaussian blur.
OpeningGray
    nn.Module wrapper for kornia.morphology.opening, with grayscale output.
OpeningBlurGray
    nn.Module wrapper for kornia.morphology.opening, with grayscale output and gaussian blur.
PointCloud
    nn.Module wrapper for kornia.geometry.depth_to_3d.
Sobel
    nn.Module wrapper for kornia.filters.sobel.
SobelBlur
    nn.Module wrapper for kornia.filters.sobel, with gaussian blur.
SobelGray
    nn.Module wrapper for kornia.filters.sobel, with grayscale output.
SobelBlurGray
    nn.Module wrapper for kornia.filters.sobel, with grayscale output and gaussian blur.
InputType
    Enum defining the type of input a model takes.
ModelType
    Enum defining the type of model.

Functions
---------
convert_to_fp16
    Use to convert an input U8 tensor to fp16.
"""

from .abstract_model import AbstractModel
from .closing import Closing, ClosingBlur, ClosingBlurGray, ClosingGray
from .dilation import Dilation, DilationBlur, DilationBlurGray, DilationGray
from .erosion import Erosion, ErosionBlur, ErosionBlurGray, ErosionGray
from .gaussian import Gaussian, GaussianGray
from .gftt import GFTT, GFTTBlur, GFTTBlurGray, GFTTGray
from .harris import Harris, HarrisBlur, HarrisBlurGray, HarrisGray
from .hessian import Hessian, HessianBlur, HessianBlurGray, HessianGray
from .laplacian import Laplacian, LaplacianBlur, LaplacianBlurGray, LaplacianGray
from .laserscan import Laserscan
from .opening import Opening, OpeningBlur, OpeningBlurGray, OpeningGray
from .point_cloud import PointCloud
from .sobel import Sobel, SobelBlur, SobelBlurGray, SobelGray
from .utils import InputType, ModelType, convert_to_fp16

__all__ = [
    "GFTT",
    "AbstractModel",
    "Closing",
    "ClosingBlur",
    "ClosingBlurGray",
    "ClosingGray",
    "Dilation",
    "DilationBlur",
    "DilationBlurGray",
    "DilationGray",
    "Erosion",
    "ErosionBlur",
    "ErosionBlurGray",
    "ErosionGray",
    "GFTTBlur",
    "GFTTBlurGray",
    "GFTTGray",
    "Gaussian",
    "GaussianGray",
    "Harris",
    "HarrisBlur",
    "HarrisBlurGray",
    "HarrisGray",
    "Hessian",
    "HessianBlur",
    "HessianBlurGray",
    "HessianGray",
    "InputType",
    "Laplacian",
    "LaplacianBlur",
    "LaplacianBlurGray",
    "LaplacianGray",
    "Laserscan",
    "ModelType",
    "Opening",
    "OpeningBlur",
    "OpeningBlurGray",
    "OpeningGray",
    "PointCloud",
    "Sobel",
    "SobelBlur",
    "SobelBlurGray",
    "SobelGray",
    "convert_to_fp16",
]
