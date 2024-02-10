# OakUtils

[![](https://img.shields.io/pypi/pyversions/oakutils.svg)](https://pypi.org/pypi/oakutils/)
![PyPI](https://img.shields.io/pypi/v/oakutils.svg?style=plastic)

![Linux](https://github.com/justincdavis/oakutils/actions/workflows/unittest-ubuntu.yaml/badge.svg?branch=main)
![Windows](https://github.com/justincdavis/oakutils/actions/workflows/unittest-windows.yaml/badge.svg?branch=main)
![MacOS](https://github.com/justincdavis/oakutils/actions/workflows/unittest-macos.yaml/badge.svg?branch=main)

![MyPy](https://github.com/justincdavis/oakutils/actions/workflows/mypy.yaml/badge.svg?branch=main)
![Ruff](https://github.com/justincdavis/oakutils/actions/workflows/ruff.yaml/badge.svg?branch=main)
![Black](https://github.com/justincdavis/oakutils/actions/workflows/black.yaml/badge.svg?branch=main)
![PyPi Build](https://github.com/justincdavis/oakutils/actions/workflows/build-check.yaml/badge.svg?branch=main)

A easy-to-use and robust library for interacting easily with OAK cameras and the DepthAI API. Aims to bridge the gap between DepthAI API and SDK allow with building in integration with OpenCV and Open3D "out-of-the-box".
Alongside this it aims to provide better tooling for creating custom functionality on the cameras and handling typical CV tasks.

---

## Documentation

https://oakutils.readthedocs.io/en/latest/

See Also:

https://pypi.org/project/oakutils/

## Installation

OakUtils is published on PyPi and which allows easy installation using pip:

`pip install oakutils`

The library can also be installed from source by first cloning and then building from source:

`git clone https://github.com/justincdavis/oakutils.git`

`cd oakutils`

`pip install .`

### Dependencies

OakUtils requires a small subset of libraries in order to function: 

* depthai
* numpy
* opencv-contrib-python
* open3d
* typing-extensions
* setuptools

## Motivation

Luxonis publishes two fantasic libraries in the form of `depthai` and `depthai-sdk`, but the level of abstraction difference between the two is large. For the average user `depthai-sdk` packages many common tools for use "out-of-the-box", but lacks the customization options. Meanwhile, `depthai` provides the lowest level access to their OAK cameras, but at the cost of verbosity. Thus, OakUtils was created to help bridge the gap between the two libraries. It was also created to allow a packaged version of tooling and functionality present in the `depthai_experiments` repo which Luxonis maintains.
An example could be the creation of the ever-present stereo depth nodes in `depthai`. The SDK allows creation in just a single line with some default parameters, but the API (`depthai`) requires up to hundreds of lines of code for all the possible customization options. The aim of OakUtils is to allow a single function call to create an object and pre-enumerate the options as parameters to the function. This will attempt to be a middle ground between the two
provided solutions.

Advantages:

* Single line calls preserved similiar to the SDK
* All parameters can be easily viewed by an end user
* Full customizability available through a function call
* Since the function calls return `depthai.Pipline` objects they are interoperable with existing users code without the user being required to use the SDK's OakCamera class creation and build within a context manager
* No dependence of a specific device or pipeline context manager

## Implementations

Currently OakUtils provides the fulling funcionality:

* aruco: Utilities for detecting, filtering, and localizing the markers and camera in the world.
* blobs: Pre-compiled models and tooling to compile your custom models and CV functions.
* calibration/CalibrationData: A submodule for quickly creating OpenCV and Open3D compatible calibration information.
* point_clouds: A submodule for easily generating and visualizing point clouds generated by from the OAK cameras data streams.
* nodes: Wrappers around DepthAI API calls to allow the easy creation of nodes.
* nodes.models: Easy to use functions for using packaged blobs.
* filters: Filters and utiltiies surrounding data at runtime.
* tools: General utilities for interacting with either the OAK or datastreams/datatypes produced by the OAK.
* Webcam: Wrapper around the API to mimic a cv2.VideoCapture
* LegacyCamera: Fixed functionality class which allows point cloud generation, 3D image generation, and more.
* APICamera: A lightweight alternative to depthai_sdk.OakCamera for processing data streams with callbacks.

## Example Usage

```python
# simple example showing how to run Sobel edge detection on the camera
import cv2
import depthai as dai

from oakutils.nodes import create_color_camera, create_xout, get_nn_bgr_frame
from oakutils.nodes.models import create_sobel

pipeline = dai.Pipeline()

# create the color camera node
cam = create_color_camera(
    pipeline, preview_size=(640, 480)
)  # set the preview size to the input of the nn

sobel = create_sobel(
    pipeline,
    input_link=cam.preview,
    shaves=1,
)
xout_sobel = create_xout(pipeline, sobel.out, "sobel")

with dai.Device(pipeline) as device:
    queue: dai.DataOutputQueue = device.getOutputQueue("sobel")

    while True:
        data = queue.get()
        frame = get_nn_bgr_frame(data, normalization=255.0)

        cv2.imshow("sobel frame", frame)
        if cv2.waitKey(1) == ord("q"):
            break
```
