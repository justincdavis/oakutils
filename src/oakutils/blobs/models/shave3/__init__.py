# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# =============================================================================
# This file is auto-generated by scripts/compile_models.py
# =============================================================================

"""
Module for 3 shave models.

Note:
----
This module is auto-generated

Attributes:
----------
Gftt : str
    nn.Module wrapper for gftt operation.
Gfttblurgray_11x11 : str
    nn.Module wrapper for gfttblurgray_11x11 operation.
Gfttblurgray_13x13 : str
    nn.Module wrapper for gfttblurgray_13x13 operation.
Gfttblurgray_15x15 : str
    nn.Module wrapper for gfttblurgray_15x15 operation.
Gfttblurgray_3x3 : str
    nn.Module wrapper for gfttblurgray_3x3 operation.
Gfttblurgray_5x5 : str
    nn.Module wrapper for gfttblurgray_5x5 operation.
Gfttblurgray_7x7 : str
    nn.Module wrapper for gfttblurgray_7x7 operation.
Gfttblurgray_9x9 : str
    nn.Module wrapper for gfttblurgray_9x9 operation.
Gfttblur_11x11 : str
    nn.Module wrapper for gfttblur_11x11 operation.
Gfttblur_13x13 : str
    nn.Module wrapper for gfttblur_13x13 operation.
Gfttblur_15x15 : str
    nn.Module wrapper for gfttblur_15x15 operation.
Gfttblur_3x3 : str
    nn.Module wrapper for gfttblur_3x3 operation.
Gfttblur_5x5 : str
    nn.Module wrapper for gfttblur_5x5 operation.
Gfttblur_7x7 : str
    nn.Module wrapper for gfttblur_7x7 operation.
Gfttblur_9x9 : str
    nn.Module wrapper for gfttblur_9x9 operation.
Gfttgray : str
    nn.Module wrapper for gfttgray operation.
Gaussiangray_11x11 : str
    nn.Module wrapper for gaussiangray_11x11 operation.
Gaussiangray_13x13 : str
    nn.Module wrapper for gaussiangray_13x13 operation.
Gaussiangray_15x15 : str
    nn.Module wrapper for gaussiangray_15x15 operation.
Gaussiangray_3x3 : str
    nn.Module wrapper for gaussiangray_3x3 operation.
Gaussiangray_5x5 : str
    nn.Module wrapper for gaussiangray_5x5 operation.
Gaussiangray_7x7 : str
    nn.Module wrapper for gaussiangray_7x7 operation.
Gaussiangray_9x9 : str
    nn.Module wrapper for gaussiangray_9x9 operation.
Gaussian_11x11 : str
    nn.Module wrapper for gaussian_11x11 operation.
Gaussian_13x13 : str
    nn.Module wrapper for gaussian_13x13 operation.
Gaussian_15x15 : str
    nn.Module wrapper for gaussian_15x15 operation.
Gaussian_3x3 : str
    nn.Module wrapper for gaussian_3x3 operation.
Gaussian_5x5 : str
    nn.Module wrapper for gaussian_5x5 operation.
Gaussian_7x7 : str
    nn.Module wrapper for gaussian_7x7 operation.
Gaussian_9x9 : str
    nn.Module wrapper for gaussian_9x9 operation.
Harris : str
    nn.Module wrapper for harris operation.
Harrisblurgray_11x11 : str
    nn.Module wrapper for harrisblurgray_11x11 operation.
Harrisblurgray_13x13 : str
    nn.Module wrapper for harrisblurgray_13x13 operation.
Harrisblurgray_15x15 : str
    nn.Module wrapper for harrisblurgray_15x15 operation.
Harrisblurgray_3x3 : str
    nn.Module wrapper for harrisblurgray_3x3 operation.
Harrisblurgray_5x5 : str
    nn.Module wrapper for harrisblurgray_5x5 operation.
Harrisblurgray_7x7 : str
    nn.Module wrapper for harrisblurgray_7x7 operation.
Harrisblurgray_9x9 : str
    nn.Module wrapper for harrisblurgray_9x9 operation.
Harrisblur_11x11 : str
    nn.Module wrapper for harrisblur_11x11 operation.
Harrisblur_13x13 : str
    nn.Module wrapper for harrisblur_13x13 operation.
Harrisblur_15x15 : str
    nn.Module wrapper for harrisblur_15x15 operation.
Harrisblur_3x3 : str
    nn.Module wrapper for harrisblur_3x3 operation.
Harrisblur_5x5 : str
    nn.Module wrapper for harrisblur_5x5 operation.
Harrisblur_7x7 : str
    nn.Module wrapper for harrisblur_7x7 operation.
Harrisblur_9x9 : str
    nn.Module wrapper for harrisblur_9x9 operation.
Harrisgray : str
    nn.Module wrapper for harrisgray operation.
Hessian : str
    nn.Module wrapper for hessian operation.
Hessianblurgray_11x11 : str
    nn.Module wrapper for hessianblurgray_11x11 operation.
Hessianblurgray_13x13 : str
    nn.Module wrapper for hessianblurgray_13x13 operation.
Hessianblurgray_15x15 : str
    nn.Module wrapper for hessianblurgray_15x15 operation.
Hessianblurgray_3x3 : str
    nn.Module wrapper for hessianblurgray_3x3 operation.
Hessianblurgray_5x5 : str
    nn.Module wrapper for hessianblurgray_5x5 operation.
Hessianblurgray_7x7 : str
    nn.Module wrapper for hessianblurgray_7x7 operation.
Hessianblurgray_9x9 : str
    nn.Module wrapper for hessianblurgray_9x9 operation.
Hessianblur_11x11 : str
    nn.Module wrapper for hessianblur_11x11 operation.
Hessianblur_13x13 : str
    nn.Module wrapper for hessianblur_13x13 operation.
Hessianblur_15x15 : str
    nn.Module wrapper for hessianblur_15x15 operation.
Hessianblur_3x3 : str
    nn.Module wrapper for hessianblur_3x3 operation.
Hessianblur_5x5 : str
    nn.Module wrapper for hessianblur_5x5 operation.
Hessianblur_7x7 : str
    nn.Module wrapper for hessianblur_7x7 operation.
Hessianblur_9x9 : str
    nn.Module wrapper for hessianblur_9x9 operation.
Hessiangray : str
    nn.Module wrapper for hessiangray operation.
Laplacianblurgray_11x11_11x11 : str
    nn.Module wrapper for laplacianblurgray_11x11_11x11 operation.
Laplacianblurgray_11x11_13x13 : str
    nn.Module wrapper for laplacianblurgray_11x11_13x13 operation.
Laplacianblurgray_11x11_15x15 : str
    nn.Module wrapper for laplacianblurgray_11x11_15x15 operation.
Laplacianblurgray_11x11_3x3 : str
    nn.Module wrapper for laplacianblurgray_11x11_3x3 operation.
Laplacianblurgray_11x11_5x5 : str
    nn.Module wrapper for laplacianblurgray_11x11_5x5 operation.
Laplacianblurgray_11x11_7x7 : str
    nn.Module wrapper for laplacianblurgray_11x11_7x7 operation.
Laplacianblurgray_11x11_9x9 : str
    nn.Module wrapper for laplacianblurgray_11x11_9x9 operation.
Laplacianblurgray_13x13_11x11 : str
    nn.Module wrapper for laplacianblurgray_13x13_11x11 operation.
Laplacianblurgray_13x13_13x13 : str
    nn.Module wrapper for laplacianblurgray_13x13_13x13 operation.
Laplacianblurgray_13x13_15x15 : str
    nn.Module wrapper for laplacianblurgray_13x13_15x15 operation.
Laplacianblurgray_13x13_3x3 : str
    nn.Module wrapper for laplacianblurgray_13x13_3x3 operation.
Laplacianblurgray_13x13_5x5 : str
    nn.Module wrapper for laplacianblurgray_13x13_5x5 operation.
Laplacianblurgray_13x13_7x7 : str
    nn.Module wrapper for laplacianblurgray_13x13_7x7 operation.
Laplacianblurgray_13x13_9x9 : str
    nn.Module wrapper for laplacianblurgray_13x13_9x9 operation.
Laplacianblurgray_15x15_11x11 : str
    nn.Module wrapper for laplacianblurgray_15x15_11x11 operation.
Laplacianblurgray_15x15_13x13 : str
    nn.Module wrapper for laplacianblurgray_15x15_13x13 operation.
Laplacianblurgray_15x15_15x15 : str
    nn.Module wrapper for laplacianblurgray_15x15_15x15 operation.
Laplacianblurgray_15x15_3x3 : str
    nn.Module wrapper for laplacianblurgray_15x15_3x3 operation.
Laplacianblurgray_15x15_5x5 : str
    nn.Module wrapper for laplacianblurgray_15x15_5x5 operation.
Laplacianblurgray_15x15_7x7 : str
    nn.Module wrapper for laplacianblurgray_15x15_7x7 operation.
Laplacianblurgray_15x15_9x9 : str
    nn.Module wrapper for laplacianblurgray_15x15_9x9 operation.
Laplacianblurgray_3x3_11x11 : str
    nn.Module wrapper for laplacianblurgray_3x3_11x11 operation.
Laplacianblurgray_3x3_13x13 : str
    nn.Module wrapper for laplacianblurgray_3x3_13x13 operation.
Laplacianblurgray_3x3_15x15 : str
    nn.Module wrapper for laplacianblurgray_3x3_15x15 operation.
Laplacianblurgray_3x3_3x3 : str
    nn.Module wrapper for laplacianblurgray_3x3_3x3 operation.
Laplacianblurgray_3x3_5x5 : str
    nn.Module wrapper for laplacianblurgray_3x3_5x5 operation.
Laplacianblurgray_3x3_7x7 : str
    nn.Module wrapper for laplacianblurgray_3x3_7x7 operation.
Laplacianblurgray_3x3_9x9 : str
    nn.Module wrapper for laplacianblurgray_3x3_9x9 operation.
Laplacianblurgray_5x5_11x11 : str
    nn.Module wrapper for laplacianblurgray_5x5_11x11 operation.
Laplacianblurgray_5x5_13x13 : str
    nn.Module wrapper for laplacianblurgray_5x5_13x13 operation.
Laplacianblurgray_5x5_15x15 : str
    nn.Module wrapper for laplacianblurgray_5x5_15x15 operation.
Laplacianblurgray_5x5_3x3 : str
    nn.Module wrapper for laplacianblurgray_5x5_3x3 operation.
Laplacianblurgray_5x5_5x5 : str
    nn.Module wrapper for laplacianblurgray_5x5_5x5 operation.
Laplacianblurgray_5x5_7x7 : str
    nn.Module wrapper for laplacianblurgray_5x5_7x7 operation.
Laplacianblurgray_5x5_9x9 : str
    nn.Module wrapper for laplacianblurgray_5x5_9x9 operation.
Laplacianblurgray_7x7_11x11 : str
    nn.Module wrapper for laplacianblurgray_7x7_11x11 operation.
Laplacianblurgray_7x7_13x13 : str
    nn.Module wrapper for laplacianblurgray_7x7_13x13 operation.
Laplacianblurgray_7x7_15x15 : str
    nn.Module wrapper for laplacianblurgray_7x7_15x15 operation.
Laplacianblurgray_7x7_3x3 : str
    nn.Module wrapper for laplacianblurgray_7x7_3x3 operation.
Laplacianblurgray_7x7_5x5 : str
    nn.Module wrapper for laplacianblurgray_7x7_5x5 operation.
Laplacianblurgray_7x7_7x7 : str
    nn.Module wrapper for laplacianblurgray_7x7_7x7 operation.
Laplacianblurgray_7x7_9x9 : str
    nn.Module wrapper for laplacianblurgray_7x7_9x9 operation.
Laplacianblurgray_9x9_11x11 : str
    nn.Module wrapper for laplacianblurgray_9x9_11x11 operation.
Laplacianblurgray_9x9_13x13 : str
    nn.Module wrapper for laplacianblurgray_9x9_13x13 operation.
Laplacianblurgray_9x9_15x15 : str
    nn.Module wrapper for laplacianblurgray_9x9_15x15 operation.
Laplacianblurgray_9x9_3x3 : str
    nn.Module wrapper for laplacianblurgray_9x9_3x3 operation.
Laplacianblurgray_9x9_5x5 : str
    nn.Module wrapper for laplacianblurgray_9x9_5x5 operation.
Laplacianblurgray_9x9_7x7 : str
    nn.Module wrapper for laplacianblurgray_9x9_7x7 operation.
Laplacianblurgray_9x9_9x9 : str
    nn.Module wrapper for laplacianblurgray_9x9_9x9 operation.
Laplacianblur_11x11_11x11 : str
    nn.Module wrapper for laplacianblur_11x11_11x11 operation.
Laplacianblur_11x11_13x13 : str
    nn.Module wrapper for laplacianblur_11x11_13x13 operation.
Laplacianblur_11x11_15x15 : str
    nn.Module wrapper for laplacianblur_11x11_15x15 operation.
Laplacianblur_11x11_3x3 : str
    nn.Module wrapper for laplacianblur_11x11_3x3 operation.
Laplacianblur_11x11_5x5 : str
    nn.Module wrapper for laplacianblur_11x11_5x5 operation.
Laplacianblur_11x11_7x7 : str
    nn.Module wrapper for laplacianblur_11x11_7x7 operation.
Laplacianblur_11x11_9x9 : str
    nn.Module wrapper for laplacianblur_11x11_9x9 operation.
Laplacianblur_13x13_11x11 : str
    nn.Module wrapper for laplacianblur_13x13_11x11 operation.
Laplacianblur_13x13_13x13 : str
    nn.Module wrapper for laplacianblur_13x13_13x13 operation.
Laplacianblur_13x13_15x15 : str
    nn.Module wrapper for laplacianblur_13x13_15x15 operation.
Laplacianblur_13x13_3x3 : str
    nn.Module wrapper for laplacianblur_13x13_3x3 operation.
Laplacianblur_13x13_5x5 : str
    nn.Module wrapper for laplacianblur_13x13_5x5 operation.
Laplacianblur_13x13_7x7 : str
    nn.Module wrapper for laplacianblur_13x13_7x7 operation.
Laplacianblur_13x13_9x9 : str
    nn.Module wrapper for laplacianblur_13x13_9x9 operation.
Laplacianblur_15x15_11x11 : str
    nn.Module wrapper for laplacianblur_15x15_11x11 operation.
Laplacianblur_15x15_13x13 : str
    nn.Module wrapper for laplacianblur_15x15_13x13 operation.
Laplacianblur_15x15_15x15 : str
    nn.Module wrapper for laplacianblur_15x15_15x15 operation.
Laplacianblur_15x15_3x3 : str
    nn.Module wrapper for laplacianblur_15x15_3x3 operation.
Laplacianblur_15x15_5x5 : str
    nn.Module wrapper for laplacianblur_15x15_5x5 operation.
Laplacianblur_15x15_7x7 : str
    nn.Module wrapper for laplacianblur_15x15_7x7 operation.
Laplacianblur_15x15_9x9 : str
    nn.Module wrapper for laplacianblur_15x15_9x9 operation.
Laplacianblur_3x3_11x11 : str
    nn.Module wrapper for laplacianblur_3x3_11x11 operation.
Laplacianblur_3x3_13x13 : str
    nn.Module wrapper for laplacianblur_3x3_13x13 operation.
Laplacianblur_3x3_15x15 : str
    nn.Module wrapper for laplacianblur_3x3_15x15 operation.
Laplacianblur_3x3_3x3 : str
    nn.Module wrapper for laplacianblur_3x3_3x3 operation.
Laplacianblur_3x3_5x5 : str
    nn.Module wrapper for laplacianblur_3x3_5x5 operation.
Laplacianblur_3x3_7x7 : str
    nn.Module wrapper for laplacianblur_3x3_7x7 operation.
Laplacianblur_3x3_9x9 : str
    nn.Module wrapper for laplacianblur_3x3_9x9 operation.
Laplacianblur_5x5_11x11 : str
    nn.Module wrapper for laplacianblur_5x5_11x11 operation.
Laplacianblur_5x5_13x13 : str
    nn.Module wrapper for laplacianblur_5x5_13x13 operation.
Laplacianblur_5x5_15x15 : str
    nn.Module wrapper for laplacianblur_5x5_15x15 operation.
Laplacianblur_5x5_3x3 : str
    nn.Module wrapper for laplacianblur_5x5_3x3 operation.
Laplacianblur_5x5_5x5 : str
    nn.Module wrapper for laplacianblur_5x5_5x5 operation.
Laplacianblur_5x5_7x7 : str
    nn.Module wrapper for laplacianblur_5x5_7x7 operation.
Laplacianblur_5x5_9x9 : str
    nn.Module wrapper for laplacianblur_5x5_9x9 operation.
Laplacianblur_7x7_11x11 : str
    nn.Module wrapper for laplacianblur_7x7_11x11 operation.
Laplacianblur_7x7_13x13 : str
    nn.Module wrapper for laplacianblur_7x7_13x13 operation.
Laplacianblur_7x7_15x15 : str
    nn.Module wrapper for laplacianblur_7x7_15x15 operation.
Laplacianblur_7x7_3x3 : str
    nn.Module wrapper for laplacianblur_7x7_3x3 operation.
Laplacianblur_7x7_5x5 : str
    nn.Module wrapper for laplacianblur_7x7_5x5 operation.
Laplacianblur_7x7_7x7 : str
    nn.Module wrapper for laplacianblur_7x7_7x7 operation.
Laplacianblur_7x7_9x9 : str
    nn.Module wrapper for laplacianblur_7x7_9x9 operation.
Laplacianblur_9x9_11x11 : str
    nn.Module wrapper for laplacianblur_9x9_11x11 operation.
Laplacianblur_9x9_13x13 : str
    nn.Module wrapper for laplacianblur_9x9_13x13 operation.
Laplacianblur_9x9_15x15 : str
    nn.Module wrapper for laplacianblur_9x9_15x15 operation.
Laplacianblur_9x9_3x3 : str
    nn.Module wrapper for laplacianblur_9x9_3x3 operation.
Laplacianblur_9x9_5x5 : str
    nn.Module wrapper for laplacianblur_9x9_5x5 operation.
Laplacianblur_9x9_7x7 : str
    nn.Module wrapper for laplacianblur_9x9_7x7 operation.
Laplacianblur_9x9_9x9 : str
    nn.Module wrapper for laplacianblur_9x9_9x9 operation.
Laplaciangray_11x11 : str
    nn.Module wrapper for laplaciangray_11x11 operation.
Laplaciangray_13x13 : str
    nn.Module wrapper for laplaciangray_13x13 operation.
Laplaciangray_15x15 : str
    nn.Module wrapper for laplaciangray_15x15 operation.
Laplaciangray_3x3 : str
    nn.Module wrapper for laplaciangray_3x3 operation.
Laplaciangray_5x5 : str
    nn.Module wrapper for laplaciangray_5x5 operation.
Laplaciangray_7x7 : str
    nn.Module wrapper for laplaciangray_7x7 operation.
Laplaciangray_9x9 : str
    nn.Module wrapper for laplaciangray_9x9 operation.
Laplacian_11x11 : str
    nn.Module wrapper for laplacian_11x11 operation.
Laplacian_13x13 : str
    nn.Module wrapper for laplacian_13x13 operation.
Laplacian_15x15 : str
    nn.Module wrapper for laplacian_15x15 operation.
Laplacian_3x3 : str
    nn.Module wrapper for laplacian_3x3 operation.
Laplacian_5x5 : str
    nn.Module wrapper for laplacian_5x5 operation.
Laplacian_7x7 : str
    nn.Module wrapper for laplacian_7x7 operation.
Laplacian_9x9 : str
    nn.Module wrapper for laplacian_9x9 operation.
Pointcloud : str
    nn.Module wrapper for pointcloud operation.
Sobel : str
    nn.Module wrapper for sobel operation.
Sobelblurgray_11x11 : str
    nn.Module wrapper for sobelblurgray_11x11 operation.
Sobelblurgray_13x13 : str
    nn.Module wrapper for sobelblurgray_13x13 operation.
Sobelblurgray_15x15 : str
    nn.Module wrapper for sobelblurgray_15x15 operation.
Sobelblurgray_3x3 : str
    nn.Module wrapper for sobelblurgray_3x3 operation.
Sobelblurgray_5x5 : str
    nn.Module wrapper for sobelblurgray_5x5 operation.
Sobelblurgray_7x7 : str
    nn.Module wrapper for sobelblurgray_7x7 operation.
Sobelblurgray_9x9 : str
    nn.Module wrapper for sobelblurgray_9x9 operation.
Sobelblur_11x11 : str
    nn.Module wrapper for sobelblur_11x11 operation.
Sobelblur_13x13 : str
    nn.Module wrapper for sobelblur_13x13 operation.
Sobelblur_15x15 : str
    nn.Module wrapper for sobelblur_15x15 operation.
Sobelblur_3x3 : str
    nn.Module wrapper for sobelblur_3x3 operation.
Sobelblur_5x5 : str
    nn.Module wrapper for sobelblur_5x5 operation.
Sobelblur_7x7 : str
    nn.Module wrapper for sobelblur_7x7 operation.
Sobelblur_9x9 : str
    nn.Module wrapper for sobelblur_9x9 operation.
Sobelgray : str
    nn.Module wrapper for sobelgray operation.

"""

from pathlib import Path

import pkg_resources

_RELATIVE_BLOB_FOLDER = Path("oakutils") / "blobs" / "models" / "shave3"
_PACKAGE_LOCATION = pkg_resources.get_distribution("oakutils").location
_BLOB_FOLDER = Path(_PACKAGE_LOCATION) / _RELATIVE_BLOB_FOLDER

GFTT = Path(Path(_BLOB_FOLDER) / "GFTT.blob").resolve()
GFTTBLURGRAY_11X11 = Path(Path(_BLOB_FOLDER) / "GFTTBlurGray_11x11.blob").resolve()
GFTTBLURGRAY_13X13 = Path(Path(_BLOB_FOLDER) / "GFTTBlurGray_13x13.blob").resolve()
GFTTBLURGRAY_15X15 = Path(Path(_BLOB_FOLDER) / "GFTTBlurGray_15x15.blob").resolve()
GFTTBLURGRAY_3X3 = Path(Path(_BLOB_FOLDER) / "GFTTBlurGray_3x3.blob").resolve()
GFTTBLURGRAY_5X5 = Path(Path(_BLOB_FOLDER) / "GFTTBlurGray_5x5.blob").resolve()
GFTTBLURGRAY_7X7 = Path(Path(_BLOB_FOLDER) / "GFTTBlurGray_7x7.blob").resolve()
GFTTBLURGRAY_9X9 = Path(Path(_BLOB_FOLDER) / "GFTTBlurGray_9x9.blob").resolve()
GFTTBLUR_11X11 = Path(Path(_BLOB_FOLDER) / "GFTTBlur_11x11.blob").resolve()
GFTTBLUR_13X13 = Path(Path(_BLOB_FOLDER) / "GFTTBlur_13x13.blob").resolve()
GFTTBLUR_15X15 = Path(Path(_BLOB_FOLDER) / "GFTTBlur_15x15.blob").resolve()
GFTTBLUR_3X3 = Path(Path(_BLOB_FOLDER) / "GFTTBlur_3x3.blob").resolve()
GFTTBLUR_5X5 = Path(Path(_BLOB_FOLDER) / "GFTTBlur_5x5.blob").resolve()
GFTTBLUR_7X7 = Path(Path(_BLOB_FOLDER) / "GFTTBlur_7x7.blob").resolve()
GFTTBLUR_9X9 = Path(Path(_BLOB_FOLDER) / "GFTTBlur_9x9.blob").resolve()
GFTTGRAY = Path(Path(_BLOB_FOLDER) / "GFTTGray.blob").resolve()
GAUSSIANGRAY_11X11 = Path(Path(_BLOB_FOLDER) / "GaussianGray_11x11.blob").resolve()
GAUSSIANGRAY_13X13 = Path(Path(_BLOB_FOLDER) / "GaussianGray_13x13.blob").resolve()
GAUSSIANGRAY_15X15 = Path(Path(_BLOB_FOLDER) / "GaussianGray_15x15.blob").resolve()
GAUSSIANGRAY_3X3 = Path(Path(_BLOB_FOLDER) / "GaussianGray_3x3.blob").resolve()
GAUSSIANGRAY_5X5 = Path(Path(_BLOB_FOLDER) / "GaussianGray_5x5.blob").resolve()
GAUSSIANGRAY_7X7 = Path(Path(_BLOB_FOLDER) / "GaussianGray_7x7.blob").resolve()
GAUSSIANGRAY_9X9 = Path(Path(_BLOB_FOLDER) / "GaussianGray_9x9.blob").resolve()
GAUSSIAN_11X11 = Path(Path(_BLOB_FOLDER) / "Gaussian_11x11.blob").resolve()
GAUSSIAN_13X13 = Path(Path(_BLOB_FOLDER) / "Gaussian_13x13.blob").resolve()
GAUSSIAN_15X15 = Path(Path(_BLOB_FOLDER) / "Gaussian_15x15.blob").resolve()
GAUSSIAN_3X3 = Path(Path(_BLOB_FOLDER) / "Gaussian_3x3.blob").resolve()
GAUSSIAN_5X5 = Path(Path(_BLOB_FOLDER) / "Gaussian_5x5.blob").resolve()
GAUSSIAN_7X7 = Path(Path(_BLOB_FOLDER) / "Gaussian_7x7.blob").resolve()
GAUSSIAN_9X9 = Path(Path(_BLOB_FOLDER) / "Gaussian_9x9.blob").resolve()
HARRIS = Path(Path(_BLOB_FOLDER) / "Harris.blob").resolve()
HARRISBLURGRAY_11X11 = Path(Path(_BLOB_FOLDER) / "HarrisBlurGray_11x11.blob").resolve()
HARRISBLURGRAY_13X13 = Path(Path(_BLOB_FOLDER) / "HarrisBlurGray_13x13.blob").resolve()
HARRISBLURGRAY_15X15 = Path(Path(_BLOB_FOLDER) / "HarrisBlurGray_15x15.blob").resolve()
HARRISBLURGRAY_3X3 = Path(Path(_BLOB_FOLDER) / "HarrisBlurGray_3x3.blob").resolve()
HARRISBLURGRAY_5X5 = Path(Path(_BLOB_FOLDER) / "HarrisBlurGray_5x5.blob").resolve()
HARRISBLURGRAY_7X7 = Path(Path(_BLOB_FOLDER) / "HarrisBlurGray_7x7.blob").resolve()
HARRISBLURGRAY_9X9 = Path(Path(_BLOB_FOLDER) / "HarrisBlurGray_9x9.blob").resolve()
HARRISBLUR_11X11 = Path(Path(_BLOB_FOLDER) / "HarrisBlur_11x11.blob").resolve()
HARRISBLUR_13X13 = Path(Path(_BLOB_FOLDER) / "HarrisBlur_13x13.blob").resolve()
HARRISBLUR_15X15 = Path(Path(_BLOB_FOLDER) / "HarrisBlur_15x15.blob").resolve()
HARRISBLUR_3X3 = Path(Path(_BLOB_FOLDER) / "HarrisBlur_3x3.blob").resolve()
HARRISBLUR_5X5 = Path(Path(_BLOB_FOLDER) / "HarrisBlur_5x5.blob").resolve()
HARRISBLUR_7X7 = Path(Path(_BLOB_FOLDER) / "HarrisBlur_7x7.blob").resolve()
HARRISBLUR_9X9 = Path(Path(_BLOB_FOLDER) / "HarrisBlur_9x9.blob").resolve()
HARRISGRAY = Path(Path(_BLOB_FOLDER) / "HarrisGray.blob").resolve()
HESSIAN = Path(Path(_BLOB_FOLDER) / "Hessian.blob").resolve()
HESSIANBLURGRAY_11X11 = Path(Path(_BLOB_FOLDER) / "HessianBlurGray_11x11.blob").resolve()
HESSIANBLURGRAY_13X13 = Path(Path(_BLOB_FOLDER) / "HessianBlurGray_13x13.blob").resolve()
HESSIANBLURGRAY_15X15 = Path(Path(_BLOB_FOLDER) / "HessianBlurGray_15x15.blob").resolve()
HESSIANBLURGRAY_3X3 = Path(Path(_BLOB_FOLDER) / "HessianBlurGray_3x3.blob").resolve()
HESSIANBLURGRAY_5X5 = Path(Path(_BLOB_FOLDER) / "HessianBlurGray_5x5.blob").resolve()
HESSIANBLURGRAY_7X7 = Path(Path(_BLOB_FOLDER) / "HessianBlurGray_7x7.blob").resolve()
HESSIANBLURGRAY_9X9 = Path(Path(_BLOB_FOLDER) / "HessianBlurGray_9x9.blob").resolve()
HESSIANBLUR_11X11 = Path(Path(_BLOB_FOLDER) / "HessianBlur_11x11.blob").resolve()
HESSIANBLUR_13X13 = Path(Path(_BLOB_FOLDER) / "HessianBlur_13x13.blob").resolve()
HESSIANBLUR_15X15 = Path(Path(_BLOB_FOLDER) / "HessianBlur_15x15.blob").resolve()
HESSIANBLUR_3X3 = Path(Path(_BLOB_FOLDER) / "HessianBlur_3x3.blob").resolve()
HESSIANBLUR_5X5 = Path(Path(_BLOB_FOLDER) / "HessianBlur_5x5.blob").resolve()
HESSIANBLUR_7X7 = Path(Path(_BLOB_FOLDER) / "HessianBlur_7x7.blob").resolve()
HESSIANBLUR_9X9 = Path(Path(_BLOB_FOLDER) / "HessianBlur_9x9.blob").resolve()
HESSIANGRAY = Path(Path(_BLOB_FOLDER) / "HessianGray.blob").resolve()
LAPLACIANBLURGRAY_11X11_11X11 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_11x11_11x11.blob").resolve()
LAPLACIANBLURGRAY_11X11_13X13 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_11x11_13x13.blob").resolve()
LAPLACIANBLURGRAY_11X11_15X15 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_11x11_15x15.blob").resolve()
LAPLACIANBLURGRAY_11X11_3X3 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_11x11_3x3.blob").resolve()
LAPLACIANBLURGRAY_11X11_5X5 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_11x11_5x5.blob").resolve()
LAPLACIANBLURGRAY_11X11_7X7 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_11x11_7x7.blob").resolve()
LAPLACIANBLURGRAY_11X11_9X9 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_11x11_9x9.blob").resolve()
LAPLACIANBLURGRAY_13X13_11X11 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_13x13_11x11.blob").resolve()
LAPLACIANBLURGRAY_13X13_13X13 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_13x13_13x13.blob").resolve()
LAPLACIANBLURGRAY_13X13_15X15 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_13x13_15x15.blob").resolve()
LAPLACIANBLURGRAY_13X13_3X3 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_13x13_3x3.blob").resolve()
LAPLACIANBLURGRAY_13X13_5X5 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_13x13_5x5.blob").resolve()
LAPLACIANBLURGRAY_13X13_7X7 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_13x13_7x7.blob").resolve()
LAPLACIANBLURGRAY_13X13_9X9 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_13x13_9x9.blob").resolve()
LAPLACIANBLURGRAY_15X15_11X11 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_15x15_11x11.blob").resolve()
LAPLACIANBLURGRAY_15X15_13X13 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_15x15_13x13.blob").resolve()
LAPLACIANBLURGRAY_15X15_15X15 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_15x15_15x15.blob").resolve()
LAPLACIANBLURGRAY_15X15_3X3 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_15x15_3x3.blob").resolve()
LAPLACIANBLURGRAY_15X15_5X5 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_15x15_5x5.blob").resolve()
LAPLACIANBLURGRAY_15X15_7X7 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_15x15_7x7.blob").resolve()
LAPLACIANBLURGRAY_15X15_9X9 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_15x15_9x9.blob").resolve()
LAPLACIANBLURGRAY_3X3_11X11 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_3x3_11x11.blob").resolve()
LAPLACIANBLURGRAY_3X3_13X13 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_3x3_13x13.blob").resolve()
LAPLACIANBLURGRAY_3X3_15X15 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_3x3_15x15.blob").resolve()
LAPLACIANBLURGRAY_3X3_3X3 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_3x3_3x3.blob").resolve()
LAPLACIANBLURGRAY_3X3_5X5 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_3x3_5x5.blob").resolve()
LAPLACIANBLURGRAY_3X3_7X7 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_3x3_7x7.blob").resolve()
LAPLACIANBLURGRAY_3X3_9X9 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_3x3_9x9.blob").resolve()
LAPLACIANBLURGRAY_5X5_11X11 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_5x5_11x11.blob").resolve()
LAPLACIANBLURGRAY_5X5_13X13 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_5x5_13x13.blob").resolve()
LAPLACIANBLURGRAY_5X5_15X15 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_5x5_15x15.blob").resolve()
LAPLACIANBLURGRAY_5X5_3X3 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_5x5_3x3.blob").resolve()
LAPLACIANBLURGRAY_5X5_5X5 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_5x5_5x5.blob").resolve()
LAPLACIANBLURGRAY_5X5_7X7 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_5x5_7x7.blob").resolve()
LAPLACIANBLURGRAY_5X5_9X9 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_5x5_9x9.blob").resolve()
LAPLACIANBLURGRAY_7X7_11X11 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_7x7_11x11.blob").resolve()
LAPLACIANBLURGRAY_7X7_13X13 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_7x7_13x13.blob").resolve()
LAPLACIANBLURGRAY_7X7_15X15 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_7x7_15x15.blob").resolve()
LAPLACIANBLURGRAY_7X7_3X3 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_7x7_3x3.blob").resolve()
LAPLACIANBLURGRAY_7X7_5X5 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_7x7_5x5.blob").resolve()
LAPLACIANBLURGRAY_7X7_7X7 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_7x7_7x7.blob").resolve()
LAPLACIANBLURGRAY_7X7_9X9 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_7x7_9x9.blob").resolve()
LAPLACIANBLURGRAY_9X9_11X11 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_9x9_11x11.blob").resolve()
LAPLACIANBLURGRAY_9X9_13X13 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_9x9_13x13.blob").resolve()
LAPLACIANBLURGRAY_9X9_15X15 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_9x9_15x15.blob").resolve()
LAPLACIANBLURGRAY_9X9_3X3 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_9x9_3x3.blob").resolve()
LAPLACIANBLURGRAY_9X9_5X5 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_9x9_5x5.blob").resolve()
LAPLACIANBLURGRAY_9X9_7X7 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_9x9_7x7.blob").resolve()
LAPLACIANBLURGRAY_9X9_9X9 = Path(Path(_BLOB_FOLDER) / "LaplacianBlurGray_9x9_9x9.blob").resolve()
LAPLACIANBLUR_11X11_11X11 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_11x11_11x11.blob").resolve()
LAPLACIANBLUR_11X11_13X13 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_11x11_13x13.blob").resolve()
LAPLACIANBLUR_11X11_15X15 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_11x11_15x15.blob").resolve()
LAPLACIANBLUR_11X11_3X3 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_11x11_3x3.blob").resolve()
LAPLACIANBLUR_11X11_5X5 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_11x11_5x5.blob").resolve()
LAPLACIANBLUR_11X11_7X7 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_11x11_7x7.blob").resolve()
LAPLACIANBLUR_11X11_9X9 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_11x11_9x9.blob").resolve()
LAPLACIANBLUR_13X13_11X11 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_13x13_11x11.blob").resolve()
LAPLACIANBLUR_13X13_13X13 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_13x13_13x13.blob").resolve()
LAPLACIANBLUR_13X13_15X15 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_13x13_15x15.blob").resolve()
LAPLACIANBLUR_13X13_3X3 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_13x13_3x3.blob").resolve()
LAPLACIANBLUR_13X13_5X5 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_13x13_5x5.blob").resolve()
LAPLACIANBLUR_13X13_7X7 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_13x13_7x7.blob").resolve()
LAPLACIANBLUR_13X13_9X9 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_13x13_9x9.blob").resolve()
LAPLACIANBLUR_15X15_11X11 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_15x15_11x11.blob").resolve()
LAPLACIANBLUR_15X15_13X13 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_15x15_13x13.blob").resolve()
LAPLACIANBLUR_15X15_15X15 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_15x15_15x15.blob").resolve()
LAPLACIANBLUR_15X15_3X3 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_15x15_3x3.blob").resolve()
LAPLACIANBLUR_15X15_5X5 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_15x15_5x5.blob").resolve()
LAPLACIANBLUR_15X15_7X7 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_15x15_7x7.blob").resolve()
LAPLACIANBLUR_15X15_9X9 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_15x15_9x9.blob").resolve()
LAPLACIANBLUR_3X3_11X11 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_3x3_11x11.blob").resolve()
LAPLACIANBLUR_3X3_13X13 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_3x3_13x13.blob").resolve()
LAPLACIANBLUR_3X3_15X15 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_3x3_15x15.blob").resolve()
LAPLACIANBLUR_3X3_3X3 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_3x3_3x3.blob").resolve()
LAPLACIANBLUR_3X3_5X5 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_3x3_5x5.blob").resolve()
LAPLACIANBLUR_3X3_7X7 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_3x3_7x7.blob").resolve()
LAPLACIANBLUR_3X3_9X9 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_3x3_9x9.blob").resolve()
LAPLACIANBLUR_5X5_11X11 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_5x5_11x11.blob").resolve()
LAPLACIANBLUR_5X5_13X13 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_5x5_13x13.blob").resolve()
LAPLACIANBLUR_5X5_15X15 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_5x5_15x15.blob").resolve()
LAPLACIANBLUR_5X5_3X3 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_5x5_3x3.blob").resolve()
LAPLACIANBLUR_5X5_5X5 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_5x5_5x5.blob").resolve()
LAPLACIANBLUR_5X5_7X7 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_5x5_7x7.blob").resolve()
LAPLACIANBLUR_5X5_9X9 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_5x5_9x9.blob").resolve()
LAPLACIANBLUR_7X7_11X11 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_7x7_11x11.blob").resolve()
LAPLACIANBLUR_7X7_13X13 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_7x7_13x13.blob").resolve()
LAPLACIANBLUR_7X7_15X15 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_7x7_15x15.blob").resolve()
LAPLACIANBLUR_7X7_3X3 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_7x7_3x3.blob").resolve()
LAPLACIANBLUR_7X7_5X5 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_7x7_5x5.blob").resolve()
LAPLACIANBLUR_7X7_7X7 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_7x7_7x7.blob").resolve()
LAPLACIANBLUR_7X7_9X9 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_7x7_9x9.blob").resolve()
LAPLACIANBLUR_9X9_11X11 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_9x9_11x11.blob").resolve()
LAPLACIANBLUR_9X9_13X13 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_9x9_13x13.blob").resolve()
LAPLACIANBLUR_9X9_15X15 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_9x9_15x15.blob").resolve()
LAPLACIANBLUR_9X9_3X3 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_9x9_3x3.blob").resolve()
LAPLACIANBLUR_9X9_5X5 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_9x9_5x5.blob").resolve()
LAPLACIANBLUR_9X9_7X7 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_9x9_7x7.blob").resolve()
LAPLACIANBLUR_9X9_9X9 = Path(Path(_BLOB_FOLDER) / "LaplacianBlur_9x9_9x9.blob").resolve()
LAPLACIANGRAY_11X11 = Path(Path(_BLOB_FOLDER) / "LaplacianGray_11x11.blob").resolve()
LAPLACIANGRAY_13X13 = Path(Path(_BLOB_FOLDER) / "LaplacianGray_13x13.blob").resolve()
LAPLACIANGRAY_15X15 = Path(Path(_BLOB_FOLDER) / "LaplacianGray_15x15.blob").resolve()
LAPLACIANGRAY_3X3 = Path(Path(_BLOB_FOLDER) / "LaplacianGray_3x3.blob").resolve()
LAPLACIANGRAY_5X5 = Path(Path(_BLOB_FOLDER) / "LaplacianGray_5x5.blob").resolve()
LAPLACIANGRAY_7X7 = Path(Path(_BLOB_FOLDER) / "LaplacianGray_7x7.blob").resolve()
LAPLACIANGRAY_9X9 = Path(Path(_BLOB_FOLDER) / "LaplacianGray_9x9.blob").resolve()
LAPLACIAN_11X11 = Path(Path(_BLOB_FOLDER) / "Laplacian_11x11.blob").resolve()
LAPLACIAN_13X13 = Path(Path(_BLOB_FOLDER) / "Laplacian_13x13.blob").resolve()
LAPLACIAN_15X15 = Path(Path(_BLOB_FOLDER) / "Laplacian_15x15.blob").resolve()
LAPLACIAN_3X3 = Path(Path(_BLOB_FOLDER) / "Laplacian_3x3.blob").resolve()
LAPLACIAN_5X5 = Path(Path(_BLOB_FOLDER) / "Laplacian_5x5.blob").resolve()
LAPLACIAN_7X7 = Path(Path(_BLOB_FOLDER) / "Laplacian_7x7.blob").resolve()
LAPLACIAN_9X9 = Path(Path(_BLOB_FOLDER) / "Laplacian_9x9.blob").resolve()
POINTCLOUD = Path(Path(_BLOB_FOLDER) / "PointCloud.blob").resolve()
SOBEL = Path(Path(_BLOB_FOLDER) / "Sobel.blob").resolve()
SOBELBLURGRAY_11X11 = Path(Path(_BLOB_FOLDER) / "SobelBlurGray_11x11.blob").resolve()
SOBELBLURGRAY_13X13 = Path(Path(_BLOB_FOLDER) / "SobelBlurGray_13x13.blob").resolve()
SOBELBLURGRAY_15X15 = Path(Path(_BLOB_FOLDER) / "SobelBlurGray_15x15.blob").resolve()
SOBELBLURGRAY_3X3 = Path(Path(_BLOB_FOLDER) / "SobelBlurGray_3x3.blob").resolve()
SOBELBLURGRAY_5X5 = Path(Path(_BLOB_FOLDER) / "SobelBlurGray_5x5.blob").resolve()
SOBELBLURGRAY_7X7 = Path(Path(_BLOB_FOLDER) / "SobelBlurGray_7x7.blob").resolve()
SOBELBLURGRAY_9X9 = Path(Path(_BLOB_FOLDER) / "SobelBlurGray_9x9.blob").resolve()
SOBELBLUR_11X11 = Path(Path(_BLOB_FOLDER) / "SobelBlur_11x11.blob").resolve()
SOBELBLUR_13X13 = Path(Path(_BLOB_FOLDER) / "SobelBlur_13x13.blob").resolve()
SOBELBLUR_15X15 = Path(Path(_BLOB_FOLDER) / "SobelBlur_15x15.blob").resolve()
SOBELBLUR_3X3 = Path(Path(_BLOB_FOLDER) / "SobelBlur_3x3.blob").resolve()
SOBELBLUR_5X5 = Path(Path(_BLOB_FOLDER) / "SobelBlur_5x5.blob").resolve()
SOBELBLUR_7X7 = Path(Path(_BLOB_FOLDER) / "SobelBlur_7x7.blob").resolve()
SOBELBLUR_9X9 = Path(Path(_BLOB_FOLDER) / "SobelBlur_9x9.blob").resolve()
SOBELGRAY = Path(Path(_BLOB_FOLDER) / "SobelGray.blob").resolve()

__all__ = [
    "GFTT",
    "GFTTBLURGRAY_11X11",
    "GFTTBLURGRAY_13X13",
    "GFTTBLURGRAY_15X15",
    "GFTTBLURGRAY_3X3",
    "GFTTBLURGRAY_5X5",
    "GFTTBLURGRAY_7X7",
    "GFTTBLURGRAY_9X9",
    "GFTTBLUR_11X11",
    "GFTTBLUR_13X13",
    "GFTTBLUR_15X15",
    "GFTTBLUR_3X3",
    "GFTTBLUR_5X5",
    "GFTTBLUR_7X7",
    "GFTTBLUR_9X9",
    "GFTTGRAY",
    "GAUSSIANGRAY_11X11",
    "GAUSSIANGRAY_13X13",
    "GAUSSIANGRAY_15X15",
    "GAUSSIANGRAY_3X3",
    "GAUSSIANGRAY_5X5",
    "GAUSSIANGRAY_7X7",
    "GAUSSIANGRAY_9X9",
    "GAUSSIAN_11X11",
    "GAUSSIAN_13X13",
    "GAUSSIAN_15X15",
    "GAUSSIAN_3X3",
    "GAUSSIAN_5X5",
    "GAUSSIAN_7X7",
    "GAUSSIAN_9X9",
    "HARRIS",
    "HARRISBLURGRAY_11X11",
    "HARRISBLURGRAY_13X13",
    "HARRISBLURGRAY_15X15",
    "HARRISBLURGRAY_3X3",
    "HARRISBLURGRAY_5X5",
    "HARRISBLURGRAY_7X7",
    "HARRISBLURGRAY_9X9",
    "HARRISBLUR_11X11",
    "HARRISBLUR_13X13",
    "HARRISBLUR_15X15",
    "HARRISBLUR_3X3",
    "HARRISBLUR_5X5",
    "HARRISBLUR_7X7",
    "HARRISBLUR_9X9",
    "HARRISGRAY",
    "HESSIAN",
    "HESSIANBLURGRAY_11X11",
    "HESSIANBLURGRAY_13X13",
    "HESSIANBLURGRAY_15X15",
    "HESSIANBLURGRAY_3X3",
    "HESSIANBLURGRAY_5X5",
    "HESSIANBLURGRAY_7X7",
    "HESSIANBLURGRAY_9X9",
    "HESSIANBLUR_11X11",
    "HESSIANBLUR_13X13",
    "HESSIANBLUR_15X15",
    "HESSIANBLUR_3X3",
    "HESSIANBLUR_5X5",
    "HESSIANBLUR_7X7",
    "HESSIANBLUR_9X9",
    "HESSIANGRAY",
    "LAPLACIANBLURGRAY_11X11_11X11",
    "LAPLACIANBLURGRAY_11X11_13X13",
    "LAPLACIANBLURGRAY_11X11_15X15",
    "LAPLACIANBLURGRAY_11X11_3X3",
    "LAPLACIANBLURGRAY_11X11_5X5",
    "LAPLACIANBLURGRAY_11X11_7X7",
    "LAPLACIANBLURGRAY_11X11_9X9",
    "LAPLACIANBLURGRAY_13X13_11X11",
    "LAPLACIANBLURGRAY_13X13_13X13",
    "LAPLACIANBLURGRAY_13X13_15X15",
    "LAPLACIANBLURGRAY_13X13_3X3",
    "LAPLACIANBLURGRAY_13X13_5X5",
    "LAPLACIANBLURGRAY_13X13_7X7",
    "LAPLACIANBLURGRAY_13X13_9X9",
    "LAPLACIANBLURGRAY_15X15_11X11",
    "LAPLACIANBLURGRAY_15X15_13X13",
    "LAPLACIANBLURGRAY_15X15_15X15",
    "LAPLACIANBLURGRAY_15X15_3X3",
    "LAPLACIANBLURGRAY_15X15_5X5",
    "LAPLACIANBLURGRAY_15X15_7X7",
    "LAPLACIANBLURGRAY_15X15_9X9",
    "LAPLACIANBLURGRAY_3X3_11X11",
    "LAPLACIANBLURGRAY_3X3_13X13",
    "LAPLACIANBLURGRAY_3X3_15X15",
    "LAPLACIANBLURGRAY_3X3_3X3",
    "LAPLACIANBLURGRAY_3X3_5X5",
    "LAPLACIANBLURGRAY_3X3_7X7",
    "LAPLACIANBLURGRAY_3X3_9X9",
    "LAPLACIANBLURGRAY_5X5_11X11",
    "LAPLACIANBLURGRAY_5X5_13X13",
    "LAPLACIANBLURGRAY_5X5_15X15",
    "LAPLACIANBLURGRAY_5X5_3X3",
    "LAPLACIANBLURGRAY_5X5_5X5",
    "LAPLACIANBLURGRAY_5X5_7X7",
    "LAPLACIANBLURGRAY_5X5_9X9",
    "LAPLACIANBLURGRAY_7X7_11X11",
    "LAPLACIANBLURGRAY_7X7_13X13",
    "LAPLACIANBLURGRAY_7X7_15X15",
    "LAPLACIANBLURGRAY_7X7_3X3",
    "LAPLACIANBLURGRAY_7X7_5X5",
    "LAPLACIANBLURGRAY_7X7_7X7",
    "LAPLACIANBLURGRAY_7X7_9X9",
    "LAPLACIANBLURGRAY_9X9_11X11",
    "LAPLACIANBLURGRAY_9X9_13X13",
    "LAPLACIANBLURGRAY_9X9_15X15",
    "LAPLACIANBLURGRAY_9X9_3X3",
    "LAPLACIANBLURGRAY_9X9_5X5",
    "LAPLACIANBLURGRAY_9X9_7X7",
    "LAPLACIANBLURGRAY_9X9_9X9",
    "LAPLACIANBLUR_11X11_11X11",
    "LAPLACIANBLUR_11X11_13X13",
    "LAPLACIANBLUR_11X11_15X15",
    "LAPLACIANBLUR_11X11_3X3",
    "LAPLACIANBLUR_11X11_5X5",
    "LAPLACIANBLUR_11X11_7X7",
    "LAPLACIANBLUR_11X11_9X9",
    "LAPLACIANBLUR_13X13_11X11",
    "LAPLACIANBLUR_13X13_13X13",
    "LAPLACIANBLUR_13X13_15X15",
    "LAPLACIANBLUR_13X13_3X3",
    "LAPLACIANBLUR_13X13_5X5",
    "LAPLACIANBLUR_13X13_7X7",
    "LAPLACIANBLUR_13X13_9X9",
    "LAPLACIANBLUR_15X15_11X11",
    "LAPLACIANBLUR_15X15_13X13",
    "LAPLACIANBLUR_15X15_15X15",
    "LAPLACIANBLUR_15X15_3X3",
    "LAPLACIANBLUR_15X15_5X5",
    "LAPLACIANBLUR_15X15_7X7",
    "LAPLACIANBLUR_15X15_9X9",
    "LAPLACIANBLUR_3X3_11X11",
    "LAPLACIANBLUR_3X3_13X13",
    "LAPLACIANBLUR_3X3_15X15",
    "LAPLACIANBLUR_3X3_3X3",
    "LAPLACIANBLUR_3X3_5X5",
    "LAPLACIANBLUR_3X3_7X7",
    "LAPLACIANBLUR_3X3_9X9",
    "LAPLACIANBLUR_5X5_11X11",
    "LAPLACIANBLUR_5X5_13X13",
    "LAPLACIANBLUR_5X5_15X15",
    "LAPLACIANBLUR_5X5_3X3",
    "LAPLACIANBLUR_5X5_5X5",
    "LAPLACIANBLUR_5X5_7X7",
    "LAPLACIANBLUR_5X5_9X9",
    "LAPLACIANBLUR_7X7_11X11",
    "LAPLACIANBLUR_7X7_13X13",
    "LAPLACIANBLUR_7X7_15X15",
    "LAPLACIANBLUR_7X7_3X3",
    "LAPLACIANBLUR_7X7_5X5",
    "LAPLACIANBLUR_7X7_7X7",
    "LAPLACIANBLUR_7X7_9X9",
    "LAPLACIANBLUR_9X9_11X11",
    "LAPLACIANBLUR_9X9_13X13",
    "LAPLACIANBLUR_9X9_15X15",
    "LAPLACIANBLUR_9X9_3X3",
    "LAPLACIANBLUR_9X9_5X5",
    "LAPLACIANBLUR_9X9_7X7",
    "LAPLACIANBLUR_9X9_9X9",
    "LAPLACIANGRAY_11X11",
    "LAPLACIANGRAY_13X13",
    "LAPLACIANGRAY_15X15",
    "LAPLACIANGRAY_3X3",
    "LAPLACIANGRAY_5X5",
    "LAPLACIANGRAY_7X7",
    "LAPLACIANGRAY_9X9",
    "LAPLACIAN_11X11",
    "LAPLACIAN_13X13",
    "LAPLACIAN_15X15",
    "LAPLACIAN_3X3",
    "LAPLACIAN_5X5",
    "LAPLACIAN_7X7",
    "LAPLACIAN_9X9",
    "POINTCLOUD",
    "SOBEL",
    "SOBELBLURGRAY_11X11",
    "SOBELBLURGRAY_13X13",
    "SOBELBLURGRAY_15X15",
    "SOBELBLURGRAY_3X3",
    "SOBELBLURGRAY_5X5",
    "SOBELBLURGRAY_7X7",
    "SOBELBLURGRAY_9X9",
    "SOBELBLUR_11X11",
    "SOBELBLUR_13X13",
    "SOBELBLUR_15X15",
    "SOBELBLUR_3X3",
    "SOBELBLUR_5X5",
    "SOBELBLUR_7X7",
    "SOBELBLUR_9X9",
    "SOBELGRAY",
]
