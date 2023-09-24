"""
Module for creating nodes for the OAK-D.

Submodules
----------
color_camera
    Module for creating color camera nodes.
image_manip
    Module for creating image manip nodes.
imu
    Module for creating imu nodes.
mono_camera
    Module for creating mono camera nodes.
neural_network
    Module for creating neural network nodes.
stereo_depth
    Module for creating stereo depth nodes.
xin
    Module for creating xin nodes.
xout
    Module for creating xout nodes.
models
    Module for creating nodes for pre-compiled models.

Functions
---------
create_color_camera
    Creates a color camera node.
create_image_manip
    Creates an image manip node.
create_imu
    Creates an imu node.
create_mono_camera
    Creates a mono camera node.
create_left_right_cameras
    Wrapper function for creating the left and right mono cameras.
create_neural_network
    Creates a neural network node.
create_stereo_depth
    Creates a stereo depth node.
create_stereo_depth_from_mono_cameras
    Creates a stereo depth node from mono cameras.
create_xin
    Creates an xin node.
create_xout
    Creates an xout node.
get_nn_frame
    Gets the output frame from the neural network node.
get_nn_bgr_frame
    Gets the output frame from the neural network node in BGR format.
get_nn_gray_frame
    Gets the output frame from the neural network node in grayscale format.
get_nn_point_cloud
    Gets the output point cloud from the neural network node.
"""
from . import (
    color_camera,
    image_manip,
    imu,
    models,
    mono_camera,
    neural_network,
    stereo_depth,
    xin,
    xout,
)
from .color_camera import create_color_camera
from .image_manip import create_image_manip
from .imu import create_imu
from .mono_camera import create_left_right_cameras, create_mono_camera
from .neural_network import (
    create_neural_network,
    get_nn_bgr_frame,
    get_nn_frame,
    get_nn_gray_frame,
    get_nn_point_cloud,
)
from .stereo_depth import create_stereo_depth, create_stereo_depth_from_mono_cameras
from .xin import create_xin
from .xout import create_xout

__all__ = [
    "color_camera",
    "image_manip",
    "imu",
    "mono_camera",
    "neural_network",
    "stereo_depth",
    "xin",
    "xout",
    "models",
    "create_color_camera",
    "create_mono_camera",
    "create_left_right_cameras",
    "create_stereo_depth",
    "create_stereo_depth_from_mono_cameras",
    "create_imu",
    "create_image_manip",
    "create_neural_network",
    "create_xout",
    "create_xin",
    "get_nn_frame",
    "get_nn_bgr_frame",
    "get_nn_gray_frame",
    "get_nn_point_cloud",
]
