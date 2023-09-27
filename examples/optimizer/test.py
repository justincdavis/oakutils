from typing import Any, Callable
from functools import partial

import depthai as dai
from oakutils.calibration import get_camera_calibration_basic
from oakutils.nodes import create_stereo_depth, create_xout, create_image_manip, create_color_camera
from oakutils.nodes.models import create_point_cloud, create_sobel
from oakutils.optimizer import Optimizer, highest_fps, lowest_avg_latency, lowest_latency


def pipeline_func(pipeline: dai.Pipeline, args: dict[str, Any]) -> list[Callable[[dai.Device], None]]:
    # generate onboard nodes
    color_cam = create_color_camera(pipeline, fps=args["color_fps"])
    color_cam.inputConfig.setBlocking(False)
    stereo, left, right = create_stereo_depth(pipeline, fps=args["mono_fps"])
    pcl, xin_pcl, start_pcl = create_point_cloud(pipeline, stereo.depth, args["calibration"], shaves=args["pcl_shaves"])
    # # create xout streams
    # xout_color = create_xout(pipeline, color_cam.preview, "color")
    manip = create_image_manip(pipeline, stereo.rectifiedLeft, frame_type=dai.RawImgFrame.Type.BGR888p, resize=(640, 480))
    manip.inputConfig.setBlocking(False)
    manip.inputConfig.setQueueSize(1)

    sobel = create_sobel(pipeline, manip.out, args["sobel_shaves"])
    xout_pcl = create_xout(pipeline, pcl.out, "pcl")
    xout_sobel = create_xout(pipeline, sobel.out, "sobel")
    # return any functions to run before starting the pipeline
    return [start_pcl]

pipeline = dai.Pipeline()
# pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_4)
funcs = pipeline_func(pipeline, {"color_fps": 30, "mono_fps": 60, "pcl_shaves": 6, "sobel_shaves": 4, "calibration": get_camera_calibration_basic()})

with dai.Device(pipeline) as device:
    for func in funcs:
        func(device)

    queues = [device.getOutputQueue(queue, maxSize=3, blocking=False) for queue in device.getOutputQueueNames()]

    iteration = 0
    while True:
        print(f"Iteration: {iteration}")
        for queue in queues:
            print(f"    Queue: {queue.getName()}")
            data = queue.get()
        iteration += 1
        