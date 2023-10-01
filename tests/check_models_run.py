from .models import check_gaussian, check_gftt, check_harris, check_hessian, check_laplacian, check_pointcloud, check_sobel


def test_gaussian():
    check_gaussian.test_gaussian_3x3_1_shave()
    check_gaussian.test_gaussian_3x3_1_shave_gray()

def test_gftt():
    check_gftt.test_gftt_3x3_1_shave()
    check_gftt.test_gftt_3x3_1_shave_gray()
    check_gftt.test_gftt_3x3_1_shave_blur()
    check_gftt.test_gftt_3x3_1_shave_blur_gray()

def test_harris():
    check_harris.test_harris_3x3_1_shave()
    check_harris.test_harris_3x3_1_shave_gray()
    check_harris.test_harris_3x3_1_shave_blur()
    check_harris.test_harris_3x3_1_shave_blur_gray()

def test_hessian():
    check_hessian.test_hessian_3x3_1_shave()
    check_hessian.test_hessian_3x3_1_shave_gray()
    check_hessian.test_hessian_3x3_1_shave_blur()
    check_hessian.test_hessian_3x3_1_shave_blur_gray()

def test_laplacian():
    check_laplacian.test_laplacian_3x3_1_shave()
    check_laplacian.test_laplacian_3x3_1_shave_gray()
    check_laplacian.test_laplacian_3x3_1_shave_blur()
    check_laplacian.test_laplacian_3x3_1_shave_blur_gray()

def test_pointcloud():
    check_pointcloud.test_pointcloud_1_shave()
    check_pointcloud.test_pointcloud_2_shave()
    check_pointcloud.test_pointcloud_3_shave()
    check_pointcloud.test_pointcloud_4_shave()
    check_pointcloud.test_pointcloud_5_shave()
    check_pointcloud.test_pointcloud_6_shave()

def test_sobel():
    check_sobel.test_sobel_3x3_1_shave()
    check_sobel.test_sobel_3x3_1_shave_gray()
    check_sobel.test_sobel_3x3_1_shave_blur()
    check_sobel.test_sobel_3x3_1_shave_blur_gray()
