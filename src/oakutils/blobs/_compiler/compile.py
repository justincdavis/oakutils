from typing import Union, Tuple, Iterable, List, Optional, Callable, Type, Dict

import torch

from ..definitions import AbstractModel
from .paths import get_cache_dir_path
from .torch import export
from .onnx import simplify
from .blob import compile_blob
from .utils import dict_to_str


def _compile(
    model: Type[AbstractModel],
    model_args: Dict,
    dummy_input_shapes: Union[Iterable[Tuple[int, int, int]], Tuple[int, int, int]],
    cache: bool = True,
) -> str:
    """
    Compiles a given torch.nn.Module class into a blob using the provided arguments.

    Parameters
    ----------
    model : AbstractModel
        The model class to compile. This should be just the type that returns an instance of the model.
        Example: `model = lambda: torchvision.models.mobilenet_v2(pretrained=True)`
        Example: `model = oakutils.blobs.definitions.GaussianBlur`
        Example: `model = oakutils.blobs.definitions.PointCloud`
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
    # make the actual model instance
    model = model(**model_args)
    input_names = model.input_names()
    output_names = model.output_names()
    arg_str = dict_to_str(model_args)

    # resolve the paths ahead of time for caching
    try:
        model_name = model.__name__
    except AttributeError:
        model_name = model.__class__.__name__
    cache_dir = get_cache_dir_path()

    onnx_path = f"{model_name}.onnx"

    # first step, export the torch model
    export(
        model_instance=model,
        dummy_input_shapes=dummy_input_shapes,
        onnx_path=onnx_path,
        input_names=input_names,
        output_names=output_names,
    )

    # second step, simplify the onnx model
    simplfiy_onnx_path = f"{model_name}_simplified.onnx"
    simplify(onnx_path, simplfiy_onnx_path)

    # third step, compile the onnx model
    compile_blob(simplfiy_onnx_path, model_name)
