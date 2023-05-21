import blobconverter


def _compile_blob(onnx_path: str, output_path: str, shaves=6):

    blobconverter.from_onnx(
        model=onnx_path,
        output_dir=output_path,
        data_type="FP16",
        shaves=shaves,
    )


def compile_blob(onnx_path: str, output_path: str, shaves=6):
    _compile_blob(onnx_path, output_path, shaves)
