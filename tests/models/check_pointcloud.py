import time
import concurrent
from typing import Any

import depthai as dai

from oakutils.calibration import get_camera_calibration
from oakutils.nodes import create_stereo_depth, create_xout, get_nn_point_cloud
from oakutils.nodes.models import create_point_cloud


TIME_TO_RUN = 10


def check_method_timout(method: callable, name: str, timeout=5) -> Any:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(method)
        try:
            result = future.result(timeout=timeout)
            assert result == 0
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError(f"{name}, timed out after {timeout} seconds")
    return result

def check_network(func: callable):
    check_method_timout(func, func.__name__, timeout=TIME_TO_RUN * 2)  # add 5 seconds to timeout to account for setup time


def check_pointcloud(shaves: int):
    """Test the sobel node"""
    pipeline = dai.Pipeline()

    calibration = get_camera_calibration(
        (1920, 1080),
        (640, 400),
        True,
    )
    stereo, left, right = create_stereo_depth(pipeline)
    pcl, xin_xyz, start_pcl = create_point_cloud(
        pipeline, 
        stereo.depth,
        calibration,
        shaves=shaves,
    )
    _ = create_xout(pipeline, pcl.out, "pcl")

    if len(dai.Device.getAllAvailableDevices()) == 0:
        return 0  # no device found
    with dai.Device(pipeline) as device:
        start_pcl(device)
        l_queue: dai.DataOutputQueue = device.getOutputQueue("pcl")

        t0 = time.perf_counter()
        while True:
            l_data = l_queue.get()
            pcl = get_nn_point_cloud(l_data)
            if time.perf_counter() - t0 > TIME_TO_RUN:
                break
    return 0

def test_pointcloud_1_shave():
    check_network(lambda: check_pointcloud(1))

def test_pointcloud_2_shave():
    check_network(lambda: check_pointcloud(2))

def test_pointcloud_3_shave():
    check_network(lambda: check_pointcloud(3))

def test_pointcloud_4_shave():
    check_network(lambda: check_pointcloud(4))

def test_pointcloud_5_shave():
    check_network(lambda: check_pointcloud(5))

def test_pointcloud_6_shave():
    check_network(lambda: check_pointcloud(6))
