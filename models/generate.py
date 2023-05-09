import itertools
from typing import Callable, Tuple
import os
import shutil
from enum import Enum

import torch
from torch import nn
import onnx
from onnxsim import simplify
import blobconverter

from definitions import Gaussian, Laplacian, Canny, Sobel, SobelBlur


class ModelType(Enum):
    """
    Enum for the different model types.
    """
    NONE = 0
    KERNEL = 1

POSSIBLE_MODELS = {
    "gaussian": (Gaussian, ModelType.KERNEL),
    "laplacian": (Laplacian, ModelType.KERNEL),
    "canny": (Canny, ModelType.KERNEL),
    "sobel": (Sobel, ModelType.NONE),
    "sobel_blur": (SobelBlur, ModelType.KERNEL),
}

# onnx folder path
ONNX_FOLDER = "models/onnx"
TEMP_ONNX_FOLDER = "models/temp_onnx"

# kernel hyperparamters
MIN_KERNEL_SIZE = 3
MAX_KERNEL_SIZE = 15
KERNEL_INCREMENT = 2

# blob folder
BLOB_FOLDER = "models/blobs"
FINAL_BLOB_FOLDER = "blobs"

def create_model_none(model: Callable) -> nn.Module:
    """
    Creates a model instance for the given model.
    """
    return model()

def create_model_kernel(model: Callable, kernel_size: int) -> nn.Module:
    """
    Creates a model instance for the given model and kernel size.
    """
    return model(kernel_size)

def create_model(
    model: Callable, model_type: int, *args, **kwargs
) -> nn.Module:
    """
    Creates a model instance for the given model and kernel size and sigma (if applicable).
    """
    try:
        if model_type == ModelType.NONE:
            return create_model_none(model)
        elif model_type == ModelType.KERNEL:
            return create_model_kernel(model, *args, **kwargs)
        else:
            raise ValueError("Invalid model type")
    except TypeError as e:
        print("Error creating model")
        raise e

def delete_folder(folder_path: str):
    """
    Deletes the folder at the given path.
    """
    if not os.path.exists(folder_path):
        return
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
    os.rmdir(folder_path)

def generate_onnx():
    # delete the onnx and temp folders
    delete_folder(ONNX_FOLDER)
    delete_folder(TEMP_ONNX_FOLDER)
    # recreate the folders
    os.makedirs(ONNX_FOLDER)
    os.makedirs(TEMP_ONNX_FOLDER)

    # load the models from the definitions
    # create a model instance for each combination of kernel size and (if applicable) sigma
    # save the models to a dictionary

    # model dict
    models = {}

    # possible kernel sizes
    kernel_sizes = [
        x for x in range(MIN_KERNEL_SIZE, MAX_KERNEL_SIZE + 1, KERNEL_INCREMENT)
    ]

    # get list of models of ModelType.NONE
    none_models = [
        model_str for model_str, (model, model_type) in POSSIBLE_MODELS.items()
        if model_type == ModelType.NONE
    ]
    for model_str in none_models:
        # create the model
        model, model_type = POSSIBLE_MODELS[model_str]
        model_instance = create_model(
            model, model_type
        )

        # store the model
        models[model_str] = model_instance

    kernel_models = [
        model_str for model_str, (model, model_type) in POSSIBLE_MODELS.items()
        if model_type == ModelType.KERNEL
    ]
    kernel_models = list(
        itertools.product(kernel_sizes, kernel_models)
    )
    for kernel_size, model_str in kernel_models:
        # create the model
        model, model_type = POSSIBLE_MODELS[model_str]
        model_instance = create_model(
            model, model_type, kernel_size
        )

        # store the model
        model_name = f"{model_str}_{kernel_size}x{kernel_size}"
        models[model_name] = model_instance

    print(f"Generated {len(models)} models.")
    # save the models in onnx format
    for model_name, model_instance in models.items():
        print(f"{model_name}: {model_instance}")

        # create dummy input
        dummy_input = torch.randn(1, 3, 480, 640)

        # onnx path
        onnx_path = os.path.join(TEMP_ONNX_FOLDER, f"{model_name}.onnx")

        # export the model
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
    
    # simplify all onnx models in the onnx folder
    model_names = os.listdir(TEMP_ONNX_FOLDER)
    for model_name in model_names:
        model_path = os.path.join(TEMP_ONNX_FOLDER, model_name)
        model = onnx.load(model_path)
        model_simp, check = simplify(model)

        assert check, f"Simplified ONNX model for: {model_name}, could not be validated"

        print(f"Model for: {model_name}, was simplified successfully")

        new_model_path = os.path.join(ONNX_FOLDER, model_name)
        onnx.save(model_simp, new_model_path)

    # delete the temp onnx folder
    delete_folder(TEMP_ONNX_FOLDER)

def generate_blobs():
    # delete the blob folder
    delete_folder(BLOB_FOLDER)
    # recreate the folder
    os.makedirs(BLOB_FOLDER)

    # for each model in the onnx folder
    model_names = os.listdir(ONNX_FOLDER)

    for model_name in model_names:
        # load the model
        model_path = os.path.join(ONNX_FOLDER, model_name)

        # convert the model to a blob
        blob_path = os.path.join(BLOB_FOLDER, model_name.split(".")[0])

        # create the folder for blob_path
        os.makedirs(blob_path)

        blobconverter.from_onnx(
            model=model_path, 
            output_dir=blob_path,
            data_type="FP16",
            shaves=5,
        )

def copy_blobs():
    # delete the final blob folder
    delete_folder(FINAL_BLOB_FOLDER)
    # recreate the folder
    os.makedirs(FINAL_BLOB_FOLDER)

    # copy the blobs to the blob folder
    # each blob is contained in a folder with its name
    # copy the blob file and rename to be directory_name.blob
    blob_names = os.listdir(BLOB_FOLDER)

    for blob_name in blob_names:
        # each blob_name is a directory containing a single blob file
        blob_path = os.path.join(BLOB_FOLDER, blob_name)
        blob_file = os.listdir(blob_path)[0]

        # copy the blob file to the final blob folder
        final_blob_path = os.path.join(FINAL_BLOB_FOLDER, f"{blob_name}.blob")

        shutil.copyfile(os.path.join(blob_path, blob_file), final_blob_path)


if __name__ == "__main__":
    generate_onnx()
    generate_blobs()
    copy_blobs()
