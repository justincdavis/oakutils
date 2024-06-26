# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import depthai as dai


@dataclass
class YolomodelData:
    confidence_threshold: float
    iou_threshold: float
    num_classes: int
    coordinate_size: int
    anchors: list[float]
    anchor_masks: dict[str, list[int]]
    spatial: bool | None = None
    depth_input_link: dai.Node.Output | None = None
    lower_depth_threshold: int = 100
    upper_depth_threshold: int = 20000
    num_inference_threads: int = 2
    num_nce_per_inference_thread: int | None = None
    num_pool_frames: int | None = None
    input_blocking: bool | None = None


@dataclass
class MobilenetData:
    confidence_threshold: float
    bounding_box_scale_factor: float = 0.5
    spatial: bool | None = None
    depth_input_link: dai.Node.Output | None = None
    lower_depth_threshold: int = 100
    upper_depth_threshold: int = 20000
    num_inference_threads: int = 2
    num_nce_per_inference_thread: int | None = None
    num_pool_frames: int | None = None
    input_blocking: bool | None = None
