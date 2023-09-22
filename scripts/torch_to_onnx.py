import argparse

import torch
import onnx
import onnxsim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--input_size", type=str, default="3,224,224")
    args = parser.parse_args()

    # Load model
    model = torch.load(args.model, map_location=torch.device("cpu"))
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, *map(int, args.input_size.split(",")))

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        do_constant_folding=True,
        verbose=True,
    )

    # Simplify ONNX
    onnx_model = onnx.load(args.output)
    onnx_model, _ = onnxsim.simplify(onnx_model, perform_optimization=True, check_n=5)
    onnx.save(onnx_model, args.output)


if __name__ == "__main__":
    main()
    