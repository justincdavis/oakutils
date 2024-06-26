# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import blobconverter  # type: ignore[import]

from oakutils.blobs.definitions.utils.types import input_type_to_str

if TYPE_CHECKING:
    from oakutils.blobs.definitions import AbstractModel

_log = logging.getLogger(__name__)


def compile_blob(
    model_type: AbstractModel,
    onnx_path: str,
    output_path: str,
    shaves: int = 6,
    version: str | None = None,
) -> None:
    """
    Compiles an ONNX model into a blob using the provided arguments.

    Parameters
    ----------
    model_type : AbstractModel
        The model class to compile. This should be just the type of the model
        being compiled.
    onnx_path : str
        The path to the ONNX model
    output_path : str
        The path to the compiled blob
    shaves : int, optional
        The number of shaves to use for the blob, by default 6
    version : str, optional
        The version of the blob to compile, by default None
        If None, 2022.1 will be used for FP16 input and 2021.4 will be used for U8 input

    """
    iop = "-iop "
    for input_name, input_type in model_type.input_names():
        type_str = input_type_to_str(input_type)
        iop += f"{input_name}:{type_str},"
    iop = iop[:-1]

    if "U8" in iop:
        if version is None:
            version = "2021.4"
        _log.debug("Compiling with U8 input")
        _log.debug(f"   shaves: {shaves}")
        _log.debug(f"   iop: {iop}")
        _log.debug(f"   version: {version}")
        blobconverter.from_onnx(
            model=onnx_path,
            output_dir=output_path,
            data_type="FP16",
            use_cache=False,
            shaves=shaves,
            optimizer_params=[],
            compile_params=[iop],
            version=version,  # change in version hack since U8 stuff is bad on 2022.1
        )
    else:
        if version is None:
            version = "2022.1"
        _log.debug("Compiling with FP16 input")
        _log.debug(f"   shaves: {shaves}")
        _log.debug(f"   version: {version}")
        blobconverter.from_onnx(
            model=onnx_path,
            output_dir=output_path,
            data_type="FP16",
            use_cache=False,
            shaves=shaves,
            version=version,  # be explicit about the version, due to above hack
        )
