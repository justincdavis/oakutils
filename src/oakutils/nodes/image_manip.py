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
Module for creating image manip nodes.

Functions
---------
create_image_manip
    Creates an image manip node.
"""
from __future__ import annotations

import depthai as dai


def create_image_manip(
    pipeline: dai.Pipeline,
    input_link: dai.Node.Output,
    frame_type: dai.RawImgFrame.Type,
    center_crop: tuple[float, float] | None = None,
    color_map: dai.Colormap | None = None,
    crop_rect: tuple[float, float, float, float] | None = None,
    crop_rotated_rect: tuple[dai.RotatedRect, bool] | None = None,
    resize: tuple[int, int] | None = None,
    resize_thumbnail: tuple[int, int, int, int, int] | None = None,
    rotation_degrees: float | None = None,
    rotation_radians: float | None = None,
    warp_border_fill_color: tuple[int, int, int] | None = None,
    warp_transform_four_points: tuple[list[dai.Point2f], bool] | None = None,
    warp_transform_matrix_3x3: list[float] | None = None,
    input_queue_size: int = 3,
    *,
    horizontal_flip: bool | None = None,
    keep_aspect_ratio: bool | None = None,
    vertical_flip: bool | None = None,
    input_reuse: bool | None = None,
    input_blocking: bool | None = None,
    input_wait_for_message: bool | None = None,
) -> dai.node.ImageManip:
    """
    Use to create an image manip node.

    Parameters
    ----------
    pipeline : dai.Pipeline
        The pipeline to add the image manip node to.
    input_link : dai.node.XLinkOut
        The input link to connect to the image manip node.
        Example: cam_rgb.preview.link
        Explicitly pass in the link as a non-called function.
    frame_type : dai.RawImgFrame.Type
        The frame type to output.
    center_crop : Optional[Tuple[float, float]], optional
        The center crop to apply, by default None
    color_map : Optional[dai.Colormap], optional
        The color map to apply, by default None
    crop_rect : Optional[Tuple[float, float, float, float]], optional
        The crop rect to apply, by default None
    crop_rotated_rect : Optional[Tuple[dai.RotatedRect, bool]], optional
        The crop rotated rect to apply, by default None
    horizontal_flip : Optional[bool], optional
        Whether to horizontally flip the image, by default None
    keep_aspect_ratio : Optional[bool], optional
        Whether to keep the aspect ratio, by default None
    resize : Optional[Tuple[int, int]], optional
        The resize to apply, by default None
    resize_thumbnail : Optional[Tuple[int, int, int, int, int]], optional
        The resize thumbnail to apply, by default None
    rotation_degrees : Optional[float], optional
        The rotation in degrees to apply, by default None
    rotation_radians : Optional[float], optional
        The rotation in radians to apply, by default None
    vertical_flip : Optional[bool], optional
        Whether to vertically flip the image, by default None
    warp_border_fill_color : Optional[Tuple[int, int, int]], optional
        The warp border fill color to apply, by default None
    warp_transform_four_points : Optional[Tuple[List[dai.Point2f], bool]], optional
        The warp transform four points to apply, by default None
    warp_transform_matrix_3x3 : Optional[List[float]], optional
        The warp transform matrix 3x3 to apply, by default None
    input_queue_size : int, optional
        The queue size of the input, by default 3
    input_reuse : Optional[bool], optional
        Whether to reuse the previous message, by default None
        If None, will be set to False
    input_blocking : Optional[bool], optional
        Whether to block the input, by default None
        If None, will be set to False
    input_wait_for_message : Optional[bool], optional
        Whether to wait for a message, by default None
        If None, will be set to False

    Returns
    -------
    dai.node.ImageManip
        The image manip node
    """
    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setFrameType(frame_type)

    if center_crop is not None:
        manip.initialConfig.setCenterCrop(*center_crop)
    if color_map is not None:
        manip.initialConfig.setColormap(color_map)
    if crop_rect is not None:
        manip.initialConfig.setCropRect(*crop_rect)
    if crop_rotated_rect is not None:
        manip.initialConfig.setCropRotatedRect(*crop_rotated_rect)
    if horizontal_flip is not None:
        manip.initialConfig.setHorizontalFlip(horizontal_flip)
    if keep_aspect_ratio is not None:
        manip.initialConfig.setKeepAspectRatio(keep_aspect_ratio)
    if resize is not None:
        manip.initialConfig.setResize(*resize)
    if resize_thumbnail is not None:
        manip.initialConfig.setResizeThumbnail(*resize_thumbnail)
    if rotation_degrees is not None:
        manip.initialConfig.setRotationDegrees(rotation_degrees)
    if rotation_radians is not None:
        manip.initialConfig.setRotationRadians(rotation_radians)
    if vertical_flip is not None:
        manip.initialConfig.setVerticalFlip(vertical_flip)
    if warp_border_fill_color is not None:
        manip.initialConfig.setWarpBorderFillColor(*warp_border_fill_color)
    if warp_transform_four_points is not None:
        manip.initialConfig.setWarpTransformFourPoints(*warp_transform_four_points)
    if warp_transform_matrix_3x3 is not None:
        manip.initialConfig.setWarpTransformMatrix3x3(warp_transform_matrix_3x3)

    input_link.link(manip.inputImage)

    if input_reuse is None:
        input_reuse = False
    if input_blocking is None:
        input_blocking = False
    if input_wait_for_message is None:
        input_wait_for_message = False

    manip.inputConfig.setQueueSize(input_queue_size)
    manip.inputConfig.setReusePreviousMessage(input_reuse)
    manip.inputConfig.setBlocking(input_blocking)
    manip.inputConfig.setWaitForMessage(input_wait_for_message)

    return manip
