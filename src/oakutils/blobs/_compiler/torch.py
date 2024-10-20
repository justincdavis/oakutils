# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import logging
from typing import Callable, Iterable

import torch

from oakutils.blobs.definitions.utils import InputType

_log = logging.getLogger(__name__)


def _create_dummy_input(
    input_shape: tuple[int, int, int],
    input_type: InputType,
    creation_func: Callable[[tuple[int, int, int, int]], torch.Tensor] = torch.rand,
) -> torch.Tensor:
    """
    Use to create a dummy input based on the input_shape.

    Parameters
    ----------
    input_shape : Tuple[int, int, int]
        The shape of the input tensor
        Should be in form width, height, channels
    input_type : InputType
        The type of the input tensor
    creation_func : Callable[[tuple[int, int, int, int]], torch.Tensor], optional
        The function to use to create the tensor, by default torch.rand
            Examples are: torch.rand, torch.randn, torch.zeros, torch.ones

    Returns
    -------
    torch.Tensor
        The dummy input tensor

    Raises
    ------
    ValueError
        If the input_shape is not in the correct form

    """
    _log.debug("Creating dummy input")

    whc = 3
    if len(input_shape) != whc:
        err_msg = "input_shape must be in form width, height, channels"
        raise ValueError(err_msg)
    if input_shape[2] not in [1, 3]:
        err_msg = "input_shape must have 1 or 3 channels"
        raise ValueError(err_msg)

    if input_type == InputType.U8:
        return creation_func(
            (1, input_shape[2], input_shape[1], input_shape[0]),
            dtype=torch.float32,  # type: ignore[call-arg]
        )
    if input_type == InputType.FP16:
        return creation_func(
            (1, input_shape[2], input_shape[1], input_shape[0]),
            dtype=torch.float32,  # type: ignore[call-arg]
        )
    if input_type == InputType.XYZ:
        return creation_func(
            (1, input_shape[1], input_shape[0], input_shape[2]),
            dtype=torch.float32,  # type: ignore[call-arg]
        )

    err_msg = f"Unknown input type: {input_type}"
    raise ValueError(err_msg)


def _create_multiple_dummy_input(
    input_shapes: Iterable[tuple[tuple[int, int, int], InputType]],
    creation_func: Callable[[tuple[int, int, int, int]], torch.Tensor] = torch.rand,
) -> list[torch.Tensor]:
    """
    Use to create a dummy input based on the input_shapes.

    Parameters
    ----------
    input_shapes : Iterable[Tuple[int, int, int]]
        The shapes of the input tensors
        Should be in form width, height, channels
    creation_func : Callable[[tuple[int, int, int, int]], torch.Tensor], optional
        The function to use to create the tensor, by default torch.rand
            Examples are: torch.rand, torch.randn, torch.zeros, torch.ones

    Returns
    -------
    Tuple[torch.Tensor, ...]
        The dummy input tensors

    """
    _log.debug("Creating multiple dummy inputs")
    return [
        _create_dummy_input(input_shape, input_type, creation_func)
        for input_shape, input_type in input_shapes
    ]


def _export_module_to_onnx(
    model_instance: torch.nn.Module,
    dummy_input: torch.Tensor | list[torch.Tensor],
    onnx_path: str,
    input_names: list[str],
    output_names: list[str],
    onnx_opset: int = 12,
    *,
    verbose: bool | None = None,
) -> None:
    """
    Use to run torch.onnx.export with the given parameters.

    Parameters
    ----------
    model_instance : torch.nn.Module
        The model instance to export
    dummy_input : Union[torch.Tensor, Tuple[torch.Tensor, ...]]
        The dummy input to use for the export
    onnx_path : str
        The path to save the onnx file to
    input_names : List[str]
        The names of the input tensors
    output_names : List[str]
        The names of the output tensors
    onnx_opset : int, optional
        The opset to use for the onnx export, by default 12
    verbose : bool, optional
        Whether to print out information about the export, by default False

    """
    if verbose is None:
        verbose = False

    if verbose:
        _log.debug(f"Exporting model to {onnx_path}")
        _log.debug(f"   Input names: {input_names}")
        _log.debug(f"   Output names: {output_names}")
        _log.debug("   Dummy input shapes:")
        for idx, dummy_input_tensor in enumerate(dummy_input):
            _log.debug(f"       Dummy input {idx} shape: {dummy_input_tensor.shape}")
        _log.debug(f"   Opset version: {onnx_opset}")

    _log.debug("Calling torch.onnx.export")
    torch.onnx.export(  # type: ignore[no-untyped-call]
        model_instance,
        tuple(dummy_input),
        onnx_path,
        export_params=True,
        opset_version=onnx_opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
    )
    _log.debug("Returned from torch.onnx.export")


def export(
    model_instance: torch.nn.Module,
    dummy_input_shapes: (
        list[tuple[tuple[int, int, int], InputType]]
        | tuple[tuple[int, int, int], InputType]
    ),
    onnx_path: str,
    input_names: list[str],
    output_names: list[str],
    creation_func: Callable[[tuple[int, int, int, int]], torch.Tensor] = torch.rand,
    onnx_opset: int = 12,
    *,
    verbose: bool | None = None,
) -> None:
    """
    Use to create dummy inputs based on the dummy_input_shapes and exports the model to onnx.

    Parameters
    ----------
    model_instance : torch.nn.Module
        The model instance to export
    dummy_input_shapes : Union[List[Tuple[int, int, int]], Tuple[int, int, int]]
        The dummy input shapes to use for the export
    onnx_path : str
        The path to save the onnx file to
    input_names : List[str]
        The names of the input tensors
    output_names : List[str]
        The names of the output tensors
    creation_func : Callable[[tuple[int, int, int, int]], torch.Tensor], optional
        The function to use to create the tensor, by default torch.rand
            Examples are: torch.rand, torch.randn, torch.zeros, torch.ones
    onnx_opset : int, optional
        The opset to use for the onnx export, by default 12
    verbose : bool, optional
        Whether to print out information about the export, by default False

    """
    if verbose is None:
        verbose = False

    if verbose:
        _log.debug("Exporting model to ONNX")
        _log.debug(f"   Dummy input shapes: {dummy_input_shapes}")
        _log.debug(f"   Dummy input shape type: {type(dummy_input_shapes)}")

    dummy_input: torch.Tensor | list[torch.Tensor] = torch.Tensor()
    if not isinstance(dummy_input_shapes, list):
        input_shape, input_type = dummy_input_shapes
        dummy_input = _create_dummy_input(input_shape, input_type, creation_func)
    else:
        dummy_input = _create_multiple_dummy_input(dummy_input_shapes, creation_func)

    if verbose:
        _log.debug("   Dummy inputs created successfully")

    _export_module_to_onnx(
        model_instance=model_instance,
        dummy_input=dummy_input,
        onnx_path=onnx_path,
        input_names=input_names,
        output_names=output_names,
        onnx_opset=onnx_opset,
        verbose=verbose,
    )
