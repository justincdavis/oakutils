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

import contextlib
import io
import json
import os
import shutil
from pathlib import Path
from typing import Callable

import torch
from requests.exceptions import HTTPError

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
    shaves: int = 6,
    creation_func: Callable = torch.rand,
    *,
    cache: bool | None = None,
) -> Path:
    """
    Compiles a given torch.nn.Module class into a blob using the provided arguments.

    Parameters
    ----------
    model_type : AbstractModel
        The model class to compile. This should be just the type that returns an
        instance of the model.
        Example: `model = lambda: torchvision.models.mobilenet_v2(pretrained=True)`
        Example: `model = oakutils.blobs.definitions.GaussianBlur`
        Example: `model = oakutils.blobs.definitions.PointCloud`
    model_args : Dict
        The arguments to pass to the model class
    dummy_input_shapes : Union[Iterable[Tuple[int, int, int]], Tuple[int, int, int]]
        The dummy input shapes to use for the export
    cache : bool, optional
        Whether or not to cache the blob, by default True
        Cache does not account for shave changes, so if you change the number of shaves
        you will need to recompile the blob with cache set to False.
    shaves : int, optional
        The number of shaves to use for the blob, by default 6
    creation_func : callable, optional
        The function to use to create the dummy input, by default torch.rand

    Returns
    -------
    Path
        The path to the compiled blob

    Raises
    ------
    RuntimeError
        If there is an error compiling the blob
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
    if not Path.exists(cache_dir):
        Path.mkdir(cache_dir, parents=True)
    onnx_cache_dir = Path(cache_dir) / "onnx"
    if not Path.exists(onnx_cache_dir):
        Path.mkdir(onnx_cache_dir, parents=True)
    simp_onnx_cache_dir = Path(cache_dir) / "simplified_onnx"
    if not Path.exists(simp_onnx_cache_dir):
        Path.mkdir(simp_onnx_cache_dir, parents=True)
    blob_cache_dir = Path(cache_dir) / "blobs"
    if not Path.exists(blob_cache_dir):
        Path.mkdir(blob_cache_dir, parents=True)

    # resolve the paths
    onnx_path = Path(onnx_cache_dir) / f"{model_name}.onnx"
    simplfiy_onnx_path = Path(simp_onnx_cache_dir) / f"{model_name}_simplified.onnx"
    blob_dir = Path(blob_cache_dir) / model_name
    final_blob_path = Path(cache_dir) / f"{model_name}.blob"

    # check if the model has been made before
    if cache and Path.exists(final_blob_path):
        return final_blob_path

    # if we are not caching, then remove the old blob
    if not cache:
        for p in [onnx_path, simplfiy_onnx_path, final_blob_path]:
            if Path.exists(p):
                Path.unlink(p)

    # first step, export the torch model
    export(
        model_instance=model,
        dummy_input_shapes=dummy_input_shapes,
        onnx_path=str(onnx_path.resolve()),
        input_names=input_names,
        output_names=output_names,
        creation_func=creation_func,
    )

    # second step, simplify the onnx model
    simplify(str(onnx_path.resolve()), str(simplfiy_onnx_path.resolve()))

    # third step, compile the onnx model
    try:
        with contextlib.redirect_stdout(io.StringIO()) as f:
            compile_blob(
                model_type,
                str(simplfiy_onnx_path.resolve()),
                str(blob_dir.resolve()),
                shaves=shaves,
            )
    except json.JSONDecodeError as err:
        base_str = "Error compiling blob. "
        base_str += "Usually this is caused by a corrupted json file. "
        base_str += "Try deleting the blobconverter cache directory and json file. "
        base_str += "Then recompile the blob."
        err_msg = base_str
        raise RuntimeError(err_msg) from err
    except HTTPError as err:
        err_dict: dict[str, str] = {}
        for line in f.getvalue().split("\n"):
            if ": " not in line:
                continue
            key, value = line.split(": ", maxsplit=1)
            key = key.replace('"', "").replace("\t", "").replace(" ", "")
            value = value[0 : len(value) - 1]
            err_dict[key] = value
        stderr = err_dict["stderr"]
        err_msg = (
            f"Error compiling blob for the OAK-D.\n  Error from OpenVINO: {stderr}"
        )
        raise RuntimeError(err_msg) from err

    # fourth step, move the blob to the cache directory
    blob_file = os.listdir(blob_dir)[0]
    return Path(str(shutil.copy(Path(blob_dir) / blob_file, final_blob_path)))


def compile_model(
    model_type: AbstractModel,
    model_args: dict,
    shaves: int = 6,
    shape_mapping: dict[InputType, tuple[int, int, int]] | None = None,
    creation_func: Callable = torch.rand,
    *,
    cache: bool | None = None,
) -> str:
    """
    Compiles a given torch.nn.Module class into a blob using the provided arguments.

    Parameters
    ----------
    model_type : AbstractModel
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
            InputType.FP16 -> (640, 480, 3)
            InputType.XYZ -> (640, 400, 3)
            InputType.U8 -> (640, 400, 1)
    creation_func: callable, optional
        The function to use to create the dummy input, by default torch.rand
          Examples are: torch.rand, torch.randn, torch.zeros, torch.ones

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
                err_msg = f"Unknown input type: {input_type}"
                raise ValueError(err_msg)
        else:
            dummy_input_shapes.append((shape_mapping[input_type], input_type))

    return str(
        _compile(
            model_type=model_type,
            model_args=model_args,
            dummy_input_shapes=dummy_input_shapes,
            cache=cache,
            shaves=shaves,
            creation_func=creation_func,
        ).resolve(),
    )
