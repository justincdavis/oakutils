from typing import Any, Callable
from functools import partial

import depthai as dai
from oakutils.calibration import get_camera_calibration_basic
from oakutils.nodes import create_stereo_depth, create_xout
from oakutils.nodes.models import create_point_cloud
from oakutils.optimizer import Optimizer, highest_fps


def pipeline_func(pipeline: dai.Pipeline, args: dict[str, Any]) -> list[Callable[[dai.Device], None]]:
    # generate onboard nodes
    stereo, left, right = create_stereo_depth(pipeline, fps=args["mono_fps"])
    pcl, xin_pcl, start_pcl = create_point_cloud(pipeline, stereo.depth, args["calibration"], shaves=args["pcl_shaves"])
    # create xout streams
    xout_pcl = create_xout(pipeline, pcl.out, "pcl")
    # return any functions to run before starting the pipeline
    return [start_pcl]

def main():
    calibration = get_camera_calibration_basic()
    optim = Optimizer(
        max_measure_time=30,
        measure_trials=3,
        warmup_cycles=5,
        stability_threshold=0.001,
        stability_length=30,
    )
    args = {
        "mono_fps": [60, 90, 120],
        "pcl_shaves": [6],
        "calibration": [calibration],
    }  # should find the highest fps + highest shave for all
    best_args_fps, fps_measurements = optim.optimize(
        pipeline_func=pipeline_func, 
        pipeline_args=args,
        objective_func=highest_fps,
    )
    print(f"Achieved {fps_measurements[0]}")
    print("Best args for highest fps:")
    print(f"{best_args_fps['mono_fps']} fps mono, {best_args_fps['pcl_shaves']} pcl shaves")

if __name__ == "__main__":
    main()