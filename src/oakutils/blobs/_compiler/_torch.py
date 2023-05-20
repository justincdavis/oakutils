from typing import Tuple

import torch


def _export_nn_module_to_onnx(model_instance: torch.nn.Module, onnx_path: str, input_shape: Tuple[int, int, int] = (3, 480, 640)):
    if input_shape[0] not in [1, 3]:
        raise ValueError("Input shape must be either (1, H, W) or (3, H, W)")

    if input_shape[0] == 1:
        input_shape = (1, input_shape[0], input_shape[1], input_shape[2])
    elif input_shape[0] == 3:
        input_shape = (1, input_shape[0], input_shape[1], input_shape[2])
        
    dummy_input = torch.randn(*input_shape)

    torch.onnx.export(
        model_instance,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
