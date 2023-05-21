from typing import Tuple, Union, List, Iterable

import torch


def _create_dummy_input(input_shape: Tuple[int, int, int]) -> torch.Tensor:
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
    """
    if len(input_shape) != 3:
        raise ValueError("input_shape must be in form width, height, channels")

    if input_shape[2] == 1:
        # if we are using a single channel, should assume that it will be grayscale
        # need to double the columns due to the way data is propagated through the pipeline
        return torch.ones(
            (1, input_shape[2], input_shape[1], input_shape[0] * 2), dtype=torch.float16
        )
    else:
        return torch.ones(
            (1, input_shape[2], input_shape[1], input_shape[0]), dtype=torch.float16
        )


def _create_multiple_dummy_input(
    input_shapes: Iterable[Tuple[int, int, int]]
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
    return tuple([_create_dummy_input(input_shape) for input_shape in input_shapes])


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
    dummy_input_shapes: Union[Iterable[Tuple[int, int, int]], Tuple[int, int, int]],
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
    dummY_input_shapes : Union[Iterable[Tuple[int, int, int]], Tuple[int, int, int]]
        The dummy input shapes to use for the export
    onnx_path : str
        The path to save the onnx file to
    input_names : List[str]
        The names of the input tensors
    output_names : List[str]
        The names of the output tensors
    """
    if isinstance(dummy_input_shapes[0], int):
        dummy_input = _create_dummy_input(dummy_input_shapes)
    else:
        dummy_input = _create_multiple_dummy_input(dummy_input_shapes)

    _export_module_to_onnx(
        model_instance, dummy_input, onnx_path, input_names, output_names
    )
