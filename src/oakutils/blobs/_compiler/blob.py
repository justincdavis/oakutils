from __future__ import annotations

from typing import TYPE_CHECKING

import blobconverter

from oakutils.blobs.definitions.utils.types import input_type_to_str

if TYPE_CHECKING:
    from oakutils.blobs.definitions import AbstractModel


def compile_blob(
    model_type: AbstractModel, onnx_path: str, output_path: str, shaves: int = 6
) -> None:
    """Compiles an ONNX model into a blob using the provided arguments.

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
    """
    iop = "-iop "
    for input_name, input_type in model_type.input_names():
        type_str = input_type_to_str(input_type)
        iop += f"{input_name}:{type_str},"
    iop = iop[:-1]

    if "U8" in iop:
        blobconverter.from_onnx(
            model=onnx_path,
            output_dir=output_path,
            data_type="FP16",
            use_cache=False,
            shaves=shaves,
            optimizer_params=[],
            compile_params=[iop],
            version="2021.4",  # change in version hack since U8 stuff is bad on 2022.1
        )
    else:
        blobconverter.from_onnx(
            model=onnx_path,
            output_dir=output_path,
            data_type="FP16",
            use_cache=False,
            shaves=shaves,
            version="2022.1",  # be explicit about the version, due to above hack
        )
