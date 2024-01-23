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
