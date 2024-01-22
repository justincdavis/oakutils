# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
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
create_yolo_detection_network
    Creates a yolo detection network node.
create_mobilenet_detection_network
    Creates a mobilenet detection network node.
get_nn_frame
    Gets the output frame from the neural network node.
get_nn_bgr_frame
    Gets the output frame from the neural network node in BGR format.
get_nn_gray_frame
    Gets the output frame from the neural network node in grayscale format.
get_nn_point_cloud_buffer
    Gets the output point cloud buffer from the neural network node.
frame_norm
    Adjusts a bounding box returned from an ImgDetection datatype to the frame size.
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
from ._misc import frame_norm
from .color_camera import create_color_camera
from .image_manip import create_image_manip
from .imu import create_imu
from .mobilenet_detection_network import create_mobilenet_detection_network
from .mono_camera import create_left_right_cameras, create_mono_camera
from .neural_network import (
    create_neural_network,
    get_nn_bgr_frame,
    get_nn_frame,
    get_nn_gray_frame,
    get_nn_point_cloud_buffer,
)
from .stereo_depth import create_stereo_depth, create_stereo_depth_from_mono_cameras
from .xin import create_xin
from .xout import create_xout
from .yolo_detection_network import create_yolo_detection_network

__all__ = [
    "color_camera",
    "create_color_camera",
    "create_image_manip",
    "create_imu",
    "create_left_right_cameras",
    "create_mobilenet_detection_network",
    "create_mono_camera",
    "create_neural_network",
    "create_stereo_depth",
    "create_stereo_depth_from_mono_cameras",
    "create_xin",
    "create_xout",
    "create_yolo_detection_network",
    "frame_norm",
    "get_nn_bgr_frame",
    "get_nn_frame",
    "get_nn_gray_frame",
    "get_nn_point_cloud_buffer",
    "image_manip",
    "imu",
    "models",
    "mono_camera",
    "neural_network",
    "stereo_depth",
    "xin",
    "xout",
]
