from typing import Tuple, Union, List, Iterable

import torch

from ..definitions.utils import InputType


def _create_dummy_input(
    input_shape: Tuple[int, int, int], input_type: InputType
) -> torch.Tensor:
    """
    Creates a dummy input based on the input_shape

    Parameters
    ----------
    input_shape : Tuple[int, int, int]
        The shape of the input tensor
        Should be in form width, height, channels

    Returns
    -------
    torch.Tensor
        The dummy input tensor

    Raises
    ------
    ValueError
        If the input_shape is not in the correct form
    """
    if len(input_shape) != 3:
        raise ValueError("input_shape must be in form width, height, channels")
    if input_shape[2] not in [1, 3]:
        raise ValueError("input_shape must have 1 or 3 channels")

    if input_type == InputType.U8:
        # if we are using a single channel, should assume that it will be grayscale
        # need to double the columns due to the way data is propagated through the pipeline
        return torch.ones(
            (1, input_shape[2], input_shape[1], input_shape[0] * 2), dtype=torch.float16
        )
    elif input_type == InputType.FP16:
        return torch.ones(
            (1, input_shape[2], input_shape[1], input_shape[0]), dtype=torch.float32
        )
    elif input_type == InputType.XYZ:
        return torch.ones(
            (1, input_shape[1], input_shape[2], 3), dtype=torch.float16
        )
    else:
        raise ValueError(f"Unknown input type: {input_type}")


def _create_multiple_dummy_input(
    input_shapes: Iterable[Tuple[Tuple[int, int, int], InputType]]
) -> Tuple[torch.Tensor, ...]:
    """
    Creates a dummy input based on the input_shapes

    Parameters
    ----------
    input_shapes : Iterable[Tuple[int, int, int]]
        The shapes of the input tensors
        Should be in form width, height, channels

    Returns
    -------
    Tuple[torch.Tensor, ...]
        The dummy input tensors
    """
    return tuple(
        [
            _create_dummy_input(input_shape, input_type)
            for input_shape, input_type in input_shapes
        ]
    )


def _export_module_to_onnx(
    model_instance: torch.nn.Module,
    dummy_input: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    onnx_path: str,
    input_names: List[str],
    output_names: List[str],
):
    """
    Runs torch.onnx.export with the given parameters

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
    """
    print(f"Exporting model to {onnx_path}")
    print(f"Input names: {input_names}")
    print(f"Output names: {output_names}")
    for dummy_input_tensor in dummy_input:
        print(f"Dummy input shape: {dummy_input_tensor.shape}")

    torch.onnx.export(
        model_instance,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
    )


def export(
    model_instance: torch.nn.Module,
    dummy_input_shapes: Union[
        List[Tuple[Tuple[int, int, int], InputType]],
        Tuple[Tuple[int, int, int], InputType],
    ],
    onnx_path: str,
    input_names: List[str],
    output_names: List[str],
):
    """
    Creates dummy inputs based on the dummy_input_shapes and exports the model to onnx

    Parameters
    ----------
    model_instance : torch.nn.Module
        The model instance to export
    dummY_input_shapes : Union[List[Tuple[int, int, int]], Tuple[int, int, int]]
        The dummy input shapes to use for the export
    onnx_path : str
        The path to save the onnx file to
    input_names : List[str]
        The names of the input tensors
    output_names : List[str]
        The names of the output tensors
    """
    if isinstance(dummy_input_shapes, tuple):
        input_shape, input_type = dummy_input_shapes
        dummy_input = _create_dummy_input(input_shape, input_type)
    else:
        dummy_input = _create_multiple_dummy_input(dummy_input_shapes)

    _export_module_to_onnx(
        model_instance, dummy_input, onnx_path, input_names, output_names
    )
