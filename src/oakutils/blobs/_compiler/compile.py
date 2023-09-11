from __future__ import annotations

import os
import shutil

from oakutils.blobs.definitions import AbstractModel, InputType

from .blob import compile_blob
from .onnx import simplify
from .paths import get_cache_dir_path
from .torch import export
from .utils import dict_to_str, remove_suffix


def _compile(
    model_type: AbstractModel,
    model_args: dict,
    dummy_input_shapes: list[tuple[tuple[int, int, int], InputType]]
    | tuple[tuple[int, int, int], InputType],
    cache: bool | None = None,
    shaves: int = 6,
) -> str:
    """Compiles a given torch.nn.Module class into a blob using the provided arguments.

    Parameters
    ----------
    model : AbstractModel
        The model class to compile. This should be just the type that returns an
        instance of the model.
        Example: `model = lambda: torchvision.models.mobilenet_v2(pretrained=True)`
        Example: `model = oakutils.blobs.definitions.GaussianBlur`
        Example: `model = oakutils.blobs.definitions.PointCloud`
    model_args : Dict
        The arguments to pass to the model class
    dummy_input_shapes : Union[Iterable[Tuple[int, int, int]], Tuple[int, int, int]]
        The dummy input shapes to use for the export
    input_names : List[str]
        The names of the input tensors
    output_names : List[str]
        The names of the output tensors
    cache : bool, optional
        Whether or not to cache the blob, by default True

    Returns
    -------
    str
        The path to the compiled blob
    """
    if cache is None:
        cache = True

    # make the actual model instance
    model = model_type(**model_args)
    input_data = model_type.input_names()
    input_names = [x[0] for x in input_data]
    output_names = model_type.output_names()
    arg_str = dict_to_str(model_args)

    # resolve the paths ahead of time for caching
    try:
        model_name = model.__name__
    except AttributeError:
        model_name = model.__class__.__name__
    model_name = remove_suffix(f"{model_name}_{arg_str}", "_")

    # handle the cache directorys
    cache_dir = get_cache_dir_path()
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    onnx_cache_dir = os.path.join(cache_dir, "onnx")
    if not os.path.exists(onnx_cache_dir):
        os.makedirs(onnx_cache_dir)
    simp_onnx_cache_dir = os.path.join(cache_dir, "simplified_onnx")
    if not os.path.exists(simp_onnx_cache_dir):
        os.makedirs(simp_onnx_cache_dir)
    blob_cache_dir = os.path.join(cache_dir, "blobs")
    if not os.path.exists(blob_cache_dir):
        os.makedirs(blob_cache_dir)

    # resolve the paths
    onnx_path = os.path.join(onnx_cache_dir, f"{model_name}.onnx")
    simplfiy_onnx_path = os.path.join(
        simp_onnx_cache_dir, f"{model_name}_simplified.onnx"
    )
    blob_dir = os.path.join(blob_cache_dir, model_name)
    final_blob_path = os.path.join(cache_dir, f"{model_name}.blob")

    # check if the model has been made before
    if cache and os.path.exists(final_blob_path):
        return final_blob_path

    # first step, export the torch model
    export(
        model_instance=model,
        dummy_input_shapes=dummy_input_shapes,
        onnx_path=onnx_path,
        input_names=input_names,
        output_names=output_names,
    )

    # second step, simplify the onnx model
    simplify(onnx_path, simplfiy_onnx_path)

    # third step, compile the onnx model
    compile_blob(model_type, simplfiy_onnx_path, blob_dir, shaves=shaves)

    # fourth step, move the blob to the cache directory
    blob_file = os.listdir(blob_dir)[0]
    return shutil.copy(os.path.join(blob_dir, blob_file), final_blob_path)


def compile_model(
    model_type: AbstractModel,
    model_args: dict,
    cache: bool | None = None,
    shaves: int = 6,
    shape_mapping: dict[InputType, tuple[int, int, int]] | None = None,
) -> str:
    """Compiles a given torch.nn.Module class into a blob using the provided arguments.

    Parameters
    ----------
    model : AbstractModel
        The model class to compile. This should be just the type that returns an
        instance of the model.
        Example: `model = lambda: torchvision.models.mobilenet_v2(pretrained=True)`
        Example: `model = oakutils.blobs.definitions.GaussianBlur`
        Example: `model = oakutils.blobs.definitions.PointCloud`
    model_args : Dict
        The arguments to pass to the model class
    cache : bool, optional
        Whether or not to cache the blob, by default True
    shaves : int, optional
        The number of shaves to use for the blob, by default 6
    shape_mapping : Optional[Dict[InputType, Tuple[int, int, int]]], optional
        The shape mapping to convert InputTypes to resolutions based on the setup
        of the camera.
        If None, then the default mapping is used, by default None
        Default mapping:
            InputType.FP16 -> (3, 480, 640)
            InputType.U8 -> (1, 400, 640)

    Returns
    -------
    str
        The path to the compiled blob
    """
    if cache is None:
        cache = True

    input_data = model_type.input_names()
    dummy_input_shapes = []
    for _, input_type in input_data:
        if shape_mapping is None:
            if input_type == InputType.FP16:
                dummy_input_shapes.append(((640, 480, 3), InputType.FP16))
            elif input_type == InputType.U8:
                dummy_input_shapes.append(((640, 400, 1), InputType.U8))
            elif input_type == InputType.XYZ:
                dummy_input_shapes.append(((640, 400, 3), InputType.XYZ))
            else:
                raise ValueError(f"Unknown input type: {input_type}")
        else:
            dummy_input_shapes.append((shape_mapping[input_type], input_type))

    return _compile(
        model_type=model_type,
        model_args=model_args,
        dummy_input_shapes=dummy_input_shapes,
        cache=cache,
        shaves=shaves,
    )
