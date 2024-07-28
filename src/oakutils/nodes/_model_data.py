# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
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


def get_yolo_data(yolo_json_path: Path | str) -> YolomodelData:
    """
    Create a YolomodelData object from a json file.

    Parameters
    ----------
    yolo_json_path : Path | str
        The path to the json file

    Returns
    -------
    YolomodelData
        The YolomodelData object

    Raises
    ------
    FileNotFoundError
        If the file does not exist

    """
    json_path = Path(yolo_json_path)
    if not json_path.exists():
        err_msg = f"File does not exist: {json_path}"
        raise FileNotFoundError(err_msg)
    with json_path.open("r") as f:
        yolo_data: dict = json.load(f)["nn_config"]["NN_specific_metadata"]

    return YolomodelData(
        confidence_threshold=yolo_data["confidence_threshold"],
        iou_threshold=yolo_data["iou_threshold"],
        num_classes=yolo_data["classes"],
        coordinate_size=yolo_data["coordinates"],
        anchors=yolo_data["anchors"],
        anchor_masks=yolo_data["anchor_masks"],
    )


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
