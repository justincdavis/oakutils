"""
Tools for working with the OAK devices.

Submodules
----------
depth
    Tools for working with depth images.
display
    Tools for displaying frames.
parsing
    Tools for switching between different formats.
pixel
    Tools for working with pixels.
spatial
    Tools for working with spatial coordinates.
transform
    Tools for working with transforms.

Classes
-------
HostSpatialsCalc
    Class for calculating spatial coordinates on the host.

Functions
---------
align_depth_to_rgb
    Use to align a depth image to an RGB image.
get_color_sensor_resolution_from_tuple
    Use to get the color sensor resolution from a tuple.
get_mono_sensor_resolution_from_tuple
    Use to get the mono sensor resolution from a tuple.
get_tuple_from_color_sensor_resolution
    Use to get a tuple from a color sensor resolution.
get_tuple_from_mono_sensor_resolution
    Use to get a tuple from a mono sensor resolution.
"""
from . import depth, display, parsing, pixel, spatial, transform
from .depth import align_depth_to_rgb
from .parsing import (
    get_color_sensor_resolution_from_tuple,
    get_mono_sensor_resolution_from_tuple,
    get_tuple_from_color_sensor_resolution,
    get_tuple_from_mono_sensor_resolution,
)
from .spatial import HostSpatialsCalc

__all__ = [
    "depth",
    "display",
    "parsing",
    "pixel",
    "spatial",
    "transform",
    "align_depth_to_rgb",
    "get_tuple_from_color_sensor_resolution",
    "get_tuple_from_mono_sensor_resolution",
    "get_color_sensor_resolution_from_tuple",
    "get_mono_sensor_resolution_from_tuple",
    "HostSpatialsCalc",
]
