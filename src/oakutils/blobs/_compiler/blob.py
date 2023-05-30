import blobconverter

from ..definitions import AbstractModel
from ..definitions.utils.types import input_type_to_str


def compile_blob(
    model_type: AbstractModel, onnx_path: str, output_path: str, shaves: int = 6
):
    """
    Compiles an ONNX model into a blob using the provided arguments.

    Parameters
    ----------
    model_type : AbstractModel
        The model class to compile. This should be just the type of the model being compiled.
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

    print(iop)

    if "U8" in iop:
        print("U8 found")
        blobconverter.from_onnx(
            model=onnx_path,
            output_dir=output_path,
            data_type="FP16",
            use_cache=False,
            shaves=shaves,
            # optimizer_params=[],
            compile_params=[iop],
        )
    else:
        blobconverter.from_onnx(
            model=onnx_path,
            output_dir=output_path,
            data_type="FP16",
            use_cache=False,
            shaves=shaves,
        )
