from typing import Tuple, Optional, List

import depthai as dai


def create_image_manip(
    pipeline: dai.Pipeline,
    input_link: dai.Node.Output,
    frame_type: dai.RawImgFrame.Type,
    stream_name: str = "image_manip",
    center_crop: Optional[Tuple[float, float]] = None,
    color_map: Optional[dai.Colormap] = None,
    crop_rect: Optional[Tuple[float, float, float, float]] = None,
    crop_rotated_rect: Optional[Tuple[dai.RotatedRect, bool]] = None,
    horizontal_flip: Optional[bool] = None,
    keep_aspect_ratio: Optional[bool] = None,
    resize: Optional[Tuple[int, int]] = None,
    resize_thumbnail: Optional[Tuple[int, int, int, int, int]] = None,
    rotation_degrees: Optional[float] = None,
    rotation_radians: Optional[float] = None,
    vertical_flip: Optional[bool] = None,
    warp_border_fill_color: Optional[Tuple[int, int, int]] = None,
    warp_transform_four_points: Optional[Tuple[List[dai.Point2f], bool]] = None,
    warp_transform_matrix_3x3: Optional[List[float]] = None,
) -> Tuple[dai.node.ImageManip, dai.node.XLinkOut]:
    """
    Creates an image manip node.

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

    Returns
    -------
    dai.node.ImageManip
        The image manip node.
    dai.node.XLinkOut
        The output link, with stream name set to the parameter given.
        Default stream name is 'image_manip'.
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
        manip.initialConfig.setWarpTransformMatrix3x3(*warp_transform_matrix_3x3)

    input_link.link(manip.inputImage)

    xout_manip = pipeline.create(dai.node.XLinkOut)
    xout_manip.setStreamName(stream_name)
    manip.out.link(xout_manip.input)

    return manip, xout_manip
