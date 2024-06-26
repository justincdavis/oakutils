# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from .test_gaussian import test_gaussian_3x3_1_shave, test_gaussian_3x3_1_shave_gray, test_gaussian_15x15_6_shave, test_gaussian_15x15_6_shave_gray
from .test_gftt import test_gftt_3x3_1_shave, test_gftt_3x3_1_shave_blur, test_gftt_3x3_1_shave_blur_gray, test_gftt_3x3_1_shave_gray
from .test_harris import test_harris_3x3_1_shave, test_harris_3x3_1_shave_blur, test_harris_3x3_1_shave_blur_gray, test_harris_3x3_1_shave_gray
from .test_hessian import test_hessian_3x3_1_shave, test_hessian_3x3_1_shave_blur, test_hessian_3x3_1_shave_blur_gray, test_hessian_3x3_1_shave_gray
from .test_laplacian import test_laplacian_3x3_1_shave, test_laplacian_3x3_1_shave_blur, test_laplacian_3x3_1_shave_blur_gray, test_laplacian_3x3_1_shave_gray
from .test_pointcloud import test_pointcloud_1_shave, test_pointcloud_2_shave, test_pointcloud_3_shave, test_pointcloud_4_shave, test_pointcloud_5_shave, test_pointcloud_6_shave
from .test_sobel import test_sobel_1_shave, test_sobel_1_shave_3x3_blur, test_sobel_1_shave_3x3_blur_gray, test_sobel_1_shave_gray

__all__ = [
    "test_gaussian_3x3_1_shave",
    "test_gaussian_3x3_1_shave_gray",
    "test_gaussian_15x15_6_shave",
    "test_gaussian_15x15_6_shave_gray",
    "test_gftt_3x3_1_shave",
    "test_gftt_3x3_1_shave_blur",
    "test_gftt_3x3_1_shave_blur_gray",
    "test_gftt_3x3_1_shave_gray",
    "test_harris_3x3_1_shave",
    "test_harris_3x3_1_shave_blur",
    "test_harris_3x3_1_shave_blur_gray",
    "test_harris_3x3_1_shave_gray",
    "test_hessian_3x3_1_shave",
    "test_hessian_3x3_1_shave_blur",
    "test_hessian_3x3_1_shave_blur_gray",
    "test_hessian_3x3_1_shave_gray",
    "test_laplacian_3x3_1_shave",
    "test_laplacian_3x3_1_shave_blur",
    "test_laplacian_3x3_1_shave_blur_gray",
    "test_laplacian_3x3_1_shave_gray",
    "test_pointcloud_1_shave",
    "test_pointcloud_2_shave",
    "test_pointcloud_3_shave",
    "test_pointcloud_4_shave",
    "test_pointcloud_5_shave",
    "test_pointcloud_6_shave",
    "test_sobel_1_shave",
    "test_sobel_1_shave_3x3_blur",
    "test_sobel_1_shave_3x3_blur_gray",
    "test_sobel_1_shave_gray",
]
