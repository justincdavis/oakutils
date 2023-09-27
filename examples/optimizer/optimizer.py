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
        "color_fps": [30],
        "mono_fps": [60],
        "pcl_shaves": [1, 6],
        "calibration": [calibration],
    }  # should find the highest fps + highest shave for all
    best_args_fps = optim.optimize(
        pipeline_func=pipeline_func, 
        pipeline_args=args,
        objective_func=highest_fps,
    )
    print("Best args for highest fps:")
    print(f"{best_args_fps['color_fps']} fps color, {best_args_fps['mono_fps']} fps mono, {best_args_fps['pcl_shaves']} pcl shaves")
    best_args_avg_latency = optim.optimize(
        pipeline_func=pipeline_func, 
        pipeline_args=args,
        objective_func=lowest_avg_latency,
    )
    print("Best args for lowest avg latency:")
    print(f"{best_args_avg_latency['color_fps']} fps color, {best_args_avg_latency['mono_fps']} fps mono, {best_args_avg_latency['pcl_shaves']} pcl shaves")
    best_args_latency = optim.optimize(
        pipeline_func=pipeline_func, 
        pipeline_args=args,
        objective_func=partial(lowest_latency, stream="pcl"),  # use partial to fill in stream name
    )
    print("Best args for lowest pcl latency:")
    print(f"{best_args_latency['color_fps']} fps color, {best_args_latency['mono_fps']} fps mono, {best_args_latency['pcl_shaves']} pcl shaves")

if __name__ == "__main__":
    main()
