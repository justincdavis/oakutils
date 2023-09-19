from __future__ import annotations

import os

import torch
import kornia
import cv2
import depthai as dai
from typing_extensions import Self
from oakutils.blobs import compile_model
from oakutils.blobs.definitions import AbstractModel, InputType, ModelType
from oakutils.nodes import create_neural_network, create_color_camera, create_xout, get_nn_bgr_frame


class Custom(AbstractModel):
    """nn.Module wrapper for a custom operation."""

    def __init__(self: Self) -> None:
        super().__init__()

    @classmethod
    def model_type(cls: Custom) -> ModelType:
        """The type of input this model takes."""
        return ModelType.KERNEL

    @classmethod
    def input_names(cls: Custom) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: Custom) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        return kornia.filters.laplacian(
            kornia.filters.gaussian_blur2d(
                image,
                (5, 5),
                (1.5, 1.5),
            ),
            7
        )
    

def main():
    model_path = compile_model(
        Custom,  # simply put the class definition here, but not a created version!
        # Sobel(),  # this is wrong!
        {},  # this model doesn't take any arguments, simply put an empty dict
        # If the model took arguments, you would put them here
        # For example, if the model took a kernel size and sigma, you would put:
        # {"kernel_size": 3, "sigma": 0.5}
        cache=False,  # this will cache the model in ~/site-packages/oakutils/blobs/_cache
        # if the model is compiled again, it will instead look in the cache for an 
        # already compiled version. Set to True to check for the model, False recompile always
        # cache=True,
        shaves=6,  # this is the number of shaves to use for the model
        # shaves are the computational units onboard the OAK cameras
        # NOTE: adding more shaves does not always mean better performance!
        # Luxonis recommends using 6 shaves for most models (actual models, not CV functions)
        # CV functions can often use 1 or 2 shaves
        shape_mapping={
            InputType.FP16: (300, 300, 3),  # this is the shape of the input tensor
            # you can change this to match whatever shapes you want your model to take as input
            # output size is determined by the model itself
            # MAKE SURE YOU DEFINE A SHAPE FOR ALL InputTypes!
            # Defaults are:
            # InputType.FP16: (640, 480, 3)
            # InputType.XYZ: (640, 400, 3)
            # InputType.U8: (640, 400, 1)
        },
        # to use default provide nothing
        # shape_mapping=None,
        creation_func=torch.ones, # this is the function used to create the "dummy" tensor
        # the dummy tensor is the data used by torch's tracer to generate the model graph
        # such that we can export it to onnx
        # the default is torch.rand, which creates a random tensor
        # you can change this to whatever you want, as long as it returns a torch.Tensor
        # Example: torch.zeros, torch.ones, torch.rand, torch.randn, torch.randperm, etc.
    )
    # model_path is the path to the compiled model
    print(model_path)

    # verify that the path exists
    assert os.path.exists(model_path)

    # verify that the path is a file
    assert os.path.isfile(model_path)

    # now lets use the new model on the camera
    pipeline = dai.Pipeline()

    # create the rgb cam to get some data
    cam = create_color_camera(
        pipeline,
        preview_size=(300, 300),  # use the preview size to get an image that matches the model
        # this is important since the resize will be done on hardware onboard the camera
        # and the normal resolution has set dimensions which do not match the models
    )
    # add the sobel model to the pipeline
    custom_network = create_neural_network(
        pipeline, 
        cam.preview,  # use the preview stream as the input
        model_path,  # our compiled model path from compile_model
    )

    # create an output stream
    streamname = "network"
    xout_nn = create_xout(pipeline, custom_network.out, streamname)

    with dai.Device(pipeline) as device:
        queue: dai.DataOutputQueue = device.getOutputQueue(streamname)

        while True:
            data = queue.get()

            # use the get_nn_bgr_frame helper to get a frame from the nn data
            # if your network doesnt output an image define a custom helper
            frame = get_nn_bgr_frame(
                data,  # the raw data packet, this will be a dai.NNData
                (300, 300),  # make sure to match the size
                normalization=255.0,  # this is how to multiply the data to get the correct values
                # by default the outputs are normalized to [0-1] by OpenVINO (the actual compiler)
                )

            cv2.imshow(streamname, frame)
            if cv2.waitKey(1) == ord("q"):
                break

if __name__ == "__main__":
    main()
