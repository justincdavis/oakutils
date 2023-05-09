import itertools
from typing import Callable, Optional, Tuple
import os
import shutil

import torch
from torch import nn
import onnx
from onnxsim import simplify
import blobconverter

from definitions import Gaussian, Laplacian


# onnx folder path
ONNX_FOLDER = "models/onnx"
TEMP_ONNX_FOLDER = "models/temp_onnx"

# kernel hyperparamters
MIN_KERNEL_SIZE = 3
MAX_KERNEL_SIZE = 11
KERNEL_INCREMENT = 2

# sigma hyperparameters
MIN_SIGMA = 1.0
MAX_SIGMA = 2.0
SIGMA_INCREMENT = 0.5

# blob folder
BLOB_FOLDER = "models/blobs"


def create_model_kernel(model: Callable, kernel_size: int) -> nn.Module:
    """
    Creates a model instance for the given model and kernel size.
    """
    return model(kernel_size)


def create_model_kernel_sigma(
    model: Callable, kernel_size: int, sigma: float
) -> nn.Module:
    """
    Creates a model instance for the given model and kernel size and sigma.
    """
    return model(kernel_size, sigma)


def create_model(
    model: Callable, kernel_size: int, sigma: Optional[float] = None
) -> Tuple[nn.Module, bool]:
    """
    Creates a model instance for the given model and kernel size and sigma (if applicable).
    """
    try:
        return create_model_kernel_sigma(model, kernel_size, sigma), True
    except TypeError:
        return create_model_kernel(model, kernel_size), False

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

    # enumerate the possible models
    possible_models = {
        "laplacian": Laplacian,
        "gaussian": Gaussian,
    }

    # model dict
    models = {}

    # possible kernel sizes
    kernel_sizes = [
        x for x in range(MIN_KERNEL_SIZE, MAX_KERNEL_SIZE + 1, KERNEL_INCREMENT)
    ]

    # possible sigmas
    sigmas = []
    num_sigmas = (MAX_SIGMA - MIN_SIGMA) // SIGMA_INCREMENT
    for i in range(int(num_sigmas) + 1):
        sigmas.append(MIN_SIGMA + i * SIGMA_INCREMENT)

    # compute the combinations of kernel_sizes and sigmas
    combinations = list(itertools.product(kernel_sizes, sigmas, possible_models.keys()))

    # create a model for each combination
    for kernel_size, sigma, model in combinations:
        # create the model
        model_instance, used_sigma = create_model(
            possible_models[model], kernel_size, sigma
        )

        # store the model
        if used_sigma:
            # convert the sigma to use an underscore instead of a decimal point
            sigma = str(sigma).replace(".", "_")
            model_name = f"{model}_{kernel_size}x{kernel_size}_{sigma}"
        else:
            model_name = f"{model}_{kernel_size}x{kernel_size}"
        models[model_name] = model_instance

    print(f"Generated {len(models)} models.")
    for model_name, model_instance in models.items():
        print(f"{model_name}: {model_instance}")

    # save the models in onnx format
    for model_name, model_instance in models.items():
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

if __name__ == "__main__":
    generate_onnx()
    generate_blobs()
