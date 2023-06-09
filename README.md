# OakUtils

A easy-to-use and robust library for interacting easily with OAK cameras and the DepthAI API. Aims to bridge the gap between DepthAI API and SDK allow with building in integration with OpenCV and Open3D "out-of-the-box".

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

`pip install .`

### Dependencies

OakUtils requires a small subset of libraries in order to function: 

depthai
numpy
opencv-python
open3d

## Motivation

Luxonis publishes two fantasic libraries in the form of `depthai` and `depthai-sdk`, but the level of abstraction difference between the two is large. For the average user `depthai-sdk` packages many common tools for use "out-of-the-box", but lacks the customization options. Meanwhile, `depthai` provides the lowest level access to their OAK cameras, but at the cost of verbosity. Thus, OakUtils was created to help bridge the gap between the two libraries.
An example could be the creation of the ever-present stereo depth nodes in `depthai`. The SDK allows creation in just a single line with some default parameters, but the API (`depthai`) requires up to hundreds of lines of code for all the possible customization options. The aim of OakUtils is to allow a single function call to create an object and pre-enumerate the options as parameters to the function. 

Advantages:

* Single line calls preserved similiar to the SDK
* All parameters can be easily viewed by an end user
* Full customizability available through a function call
* Since the function calls return `depthai.Pipline` objects they are interoperable with existing users code without the user being required to use the SDK's OakCamera class creation and build within a context manager
* No dependence of a specific device or pipeline context manager

Over time, not just the node creation steps, but also calibration, point clouds, pre-compiled cv functions (for the VPU), and algorithms from the depthai-experiments library will be added as easy to use functions and classes. 

## Implementations

Currently OakUtils provides the fulling funcionality:

* Camera: An easy to use abstraction over the entire camera covering the color camera, stereo depth, and the imu. This also includes integrations with the calibration and point cloud submodules.
* calibration/CalibrationData: A submodule for quickly creating OpenCV compatible calibration information
* point_clouds: A submodule for easily generating and visualizing point clouds generated by from the OAK cameras data streams.

## Usage

```python
from oakutils import Camera
oak_cam = Camera(
    compute_point_cloud_on_demand=False,  # always compute point cloud on data recevied
    display_point_clouds=True,  # display point clouds in display thread
    display_depth=True,  # display the depth data
)
oak_cam.start(block=True)
# after this point the camera is running in a background thread
import time
time.sleep(10)  # this is a placeholder, your program would go here (as an example)
oak_cam.stop()
```
