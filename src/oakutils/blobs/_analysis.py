# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import depthai as dai


@dataclass(frozen=True)
class LayerData:
    """Dataclass for storing input/output layer data for blobs."""

    name: str
    info: dai.TensorInfo
    datatype: dai.TensorInfo.DataType
    shape: list[int]
    order: dai.TensorInfo.StorageOrder


def get_blob(blob_path: Path | str) -> dai.OpenVINO.Blob:
    """
    Load a blob from a file.

    Parameters
    ----------
    blob_path : Path | str
        The path to the blob.

    Returns
    -------
    dai.OpenVINO.Blob
        The blob object.

    Raises
    ------
    FileNotFoundError
        If the blob does not exist.

    """
    if isinstance(blob_path, str):
        blob_path = Path(blob_path)
    if not blob_path.exists():
        err_msg = f"Blob {blob_path} does not exist."
        raise FileNotFoundError(err_msg)
    return dai.OpenVINO.Blob(blob_path)


def get_input_layer_data(blob_path: Path | str) -> list[LayerData]:
    """
    Get the input layer data for a blob.

    Parameters
    ----------
    blob_path : Path | str
        The path to the blob.

    Returns
    -------
    list[LayerData]
        A list of LayerData objects.

    """
    blob = get_blob(blob_path)
    return [
        LayerData(name, vec, vec.dataType, vec.dims, vec.order)
        for name, vec in blob.networkInputs.items()
    ]


def get_output_layer_data(blob_path: Path | str) -> list[LayerData]:
    """
    Get the output layer data for a blob.

    Parameters
    ----------
    blob_path : Path | str
        The path to the blob.

    Returns
    -------
    list[LayerData]
        A list of LayerData objects.

    """
    blob = get_blob(blob_path)
    return [
        LayerData(name, vec, vec.dataType, vec.dims, vec.order)
        for name, vec in blob.networkOutputs.items()
    ]


def get_layer_data(blob_path: Path | str) -> tuple[list[LayerData], list[LayerData]]:
    """
    Get the input and output layer data for a blob.

    Parameters
    ----------
    blob_path : Path | str
        The path to the blob.

    Returns
    -------
    tuple[list[LayerData], list[LayerData]]
        A tuple of lists of LayerData objects.

    """
    return get_input_layer_data(blob_path), get_output_layer_data(blob_path)
