from typing import Any, Callable
from functools import partial

import depthai as dai
from oakutils.calibration import get_camera_calibration_basic
from oakutils.nodes import create_color_camera, create_stereo_depth, create_xout
from oakutils.nodes.models import create_point_cloud
from oakutils.optimizer import Optimizer, highest_fps, lowest_avg_latency, lowest_latency


def pipeline_func(pipeline: dai.Pipeline, args: dict[str, Any]) -> list[Callable[[dai.Device], None]]:
    # generate onboard nodes
    color_cam = create_color_camera(pipeline, fps=args["color_fps"])
    stereo, left, right = create_stereo_depth(pipeline, fps=args["mono_fps"])
    pcl, xin_pcl, start_pcl = create_point_cloud(pipeline, stereo.depth, args["calibration"], shaves=args["pcl_shaves"])
    # create xout streams
    xout_color = create_xout(pipeline, color_cam.preview, "color")
    xout_pcl = create_xout(pipeline, pcl.out, "pcl")
    # return any functions to run before starting the pipeline
    return [start_pcl]

def main():
    calibration = get_camera_calibration_basic()
    optim = Optimizer()
    args = {
        "color_fps": [10, 15, 20, 25, 30],
        "mono_fps": [30, 40, 50, 60],
        "pcl_shaves": [1, 2, 3, 4, 5, 6],
        "calibration": [calibration],
    }
    best_args_fps = optim.optimize(
        pipeline_func=pipeline_func, 
        pipeline_args=args,
        objective_func=highest_fps,
    )
    print("Best args for highest fps:", best_args_fps)
    best_args_avg_latency = optim.optimize(
        pipeline_func=pipeline_func, 
        pipeline_args=args,
        objective_func=lowest_avg_latency,
    )
    print("Best args for lowest avg latency:", best_args_avg_latency)
    best_args_latency = optim.optimize(
        pipeline_func=pipeline_func, 
        pipeline_args=args,
        objective_func=partial(lowest_latency, stream="pcl"),  # use partial to fill in stream name
    )
    print("Best args for lowest pcl latency:", best_args_latency)

if __name__ == "__main__":
    main()
