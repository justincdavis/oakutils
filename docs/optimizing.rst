.. _optimizing:

Optimizing Custom Pipelines
---------------------------

Oakutils allows you to optimize custom defined pipelines automatically. The only requirements
are that the pipeline must be able to be defined by a single function call. It should
be noted that the function must return a list of callables (more of that down below), it can be
defined arbitraryily inside of the function as long as the pipeline passed in the function is 
used. 

For example: 

.. code-block:: python

    def create_pipeline(pipeline, dict_args):
        your_create_pipeline_function(pipeline, dict_args)
        return []  # more on this later

    def your_create_pipeline_function(pipeline, dict_args):
        # do stuff with pipeline and dict_args
        # dai.someCall(pipeline, ...)
        # dai.someOtherCall(pipeline, ...)
        # etc...

The function must return a list of callables. These callables are the functions that will be
called once the pipeline is built and will have the device passed in as an argument. This exists
because some pipeline setup (i.e. the oakutils point cloud) requires secondary setup to pass data
from the host to the device post pipeline build. The behavior of these callables are up to the 
users implementation.

An example of the point cloud device call:

.. code-block:: python

    from functools import partial

    def _start_point_cloud(device: dai.Device, xyz: np.ndarray) -> None:
        buff = dai.Buffer()
        buff.setData(xyz)
        device.getInputQueue(input_stream_name).send(buff)

    device_call = partial(_start_point_cloud, xyz=xyz_data)

The any device call is assumed to have no return value so if you want to return a value
you should pre-pass a class/data structure to the device call and have the device call
modify the data structure.

The above device_call example is used in the following manner internally:

.. code-block:: python

    with dai.Device(pipeline) as device:
        # run any device calls
        for device_call in device_calls:  # device calls is the list returned from the pipeline creation func
            # passes the xyz_data from host to device and adds it to the point cloud generator
            device_call(device)

This allows arbitrary setup to be done on the device after the pipeline is built. Enabling
some more complex pipelines to be built and optimized (such a on-device point cloud generation).

Once you have a pipeline creation function which takes a pipeline and a dict of arguments
and returns a list of device calls, you are ready to begin the optimization process.

Lets look at the example provided:

.. code-block:: python

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

The above example is a simple pipeline which creates a stereo depth node, a point cloud node, and
a xout node. The pipeline function returns a list of device calls which will be called after the
pipeline is built. In this case, the device call is used to pass the calibration data to the point
cloud node. The pipeline function also takes a dict of arguments which will be used to create
the pipeline. Inside the optimizer the set of all possible argument combinations is created. This
state space is then explored via the chosen method of the optimizer. In this case, the optimizer
is using the grid search algorithm to find the true best argument combination. Other algorithms
will be added in the future. 

The objective function is what the optimizer uses to determine the best argument combination. In
this case, the objective function is highest_fps. This function is a simple function which returns
the fps to iterate through all possible xout streams. Other built-in objective functions are the 
lowest_avg_latency and lowest_latency functions. Custom objective functions can also be defined as
long as the function signature is the same as the built-in objective functions. 

An example of the lowest_avg_latency function signature is given:

.. code-block:: python

    def lowest_avg_latency(
        options: list[tuple[tuple[float, float, dict[str, float]], dict[str, Any]]]
    ) -> tuple[dict[str, Any], tuple[float, float, dict[str, float]]]:

Using these tools you can easily optimize your custom pipelines to find the best argument combination
for your use case.
