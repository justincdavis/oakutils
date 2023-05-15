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
    # "canny": (Canny, ModelType.KERNEL),
    "sobel": (Sobel, ModelType.NONE),
    "sobel_blur": (SobelBlur, ModelType.KERNEL),
}

# onnx folder path
ONNX_FOLDER = os.path.join("models", "onnx")
TEMP_ONNX_FOLDER = os.path.join("models", "temp_onnx")

# kernel hyperparamters
MIN_KERNEL_SIZE = 3
MAX_KERNEL_SIZE = 15
KERNEL_INCREMENT = 2

# blob folder
BLOB_FOLDER = os.path.join("models", "blobs")
FINAL_BLOB_FOLDER = os.path.join("src", "oakutils", "blobs")

# for init file creation
INIT_FILE = os.path.join("src", "oakutils", "blobs", "__init__.py")

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
    delete_folder(TEMP_ONNX_FOLDER)
    # recreate the folders
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

        # also create the final onnx path
        final_onnx_path = os.path.join(ONNX_FOLDER, f"{model_name}.onnx")

        # if the final onnx path already exists skip this
        if os.path.exists(final_onnx_path):
            continue

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
    
    # perform a check that the onnx folder exists, if not make it
    if not os.path.exists(ONNX_FOLDER):
        os.makedirs(ONNX_FOLDER)

    # simplify all onnx models in the onnx folder
    model_names = os.listdir(TEMP_ONNX_FOLDER)
    for model_name in model_names:
        model_path = os.path.join(TEMP_ONNX_FOLDER, model_name)
        new_model_path = os.path.join(ONNX_FOLDER, model_name)

        # if the new_model_path already exists skip this
        if os.path.exists(new_model_path):
            continue

        model = onnx.load(model_path)
        model_simp, check = simplify(model, check_n=5, perform_optimization=True)

        assert check, f"Simplified ONNX model for: {model_name}, could not be validated"

        print(f"Model for: {model_name}, was simplified successfully")

        onnx.save(model_simp, new_model_path)

    # delete the temp onnx folder
    delete_folder(TEMP_ONNX_FOLDER)

def generate_blobs():
    # delete the final blob folder
    delete_folder(FINAL_BLOB_FOLDER)
    # recreate the folder
    os.makedirs(FINAL_BLOB_FOLDER)

    # for each model in the onnx folder
    model_names = os.listdir(ONNX_FOLDER)

    for model_name in model_names:
        # load the model
        model_path = os.path.join(ONNX_FOLDER, model_name)

        # convert the model to a blob
        blob_path = os.path.join(BLOB_FOLDER, model_name.split(".")[0])

        # check if the blob folder exists
        if os.path.exists(blob_path):
            # skip the model
            pass
        else:
            # create the folder for blob_path
            os.makedirs(blob_path)

            blobconverter.from_onnx(
                model=model_path, 
                output_dir=blob_path,
                data_type="FP16",
                shaves=6,
            )
    
        blob_name = model_name.split(".")[0]

        # each blob_name is a directory containing a single blob file
        blob_path = os.path.join(BLOB_FOLDER, blob_name)
        blob_file = os.listdir(blob_path)[0]

        # copy the blob file to the final blob folder
        final_blob_path = os.path.join(FINAL_BLOB_FOLDER, f"_{blob_name}.blob")

        shutil.copyfile(os.path.join(blob_path, blob_file), final_blob_path)
        
def create_init():
    # create an __init__.py file which will contain the model names
    # this allows easy importing of the blob paths as pre-defined variables
    # in the __init__.py file
    
    # delete the init file
    try:
        os.remove(INIT_FILE)
    except FileNotFoundError:
        pass

    # store the names of all defined variables for making the __all__
    var_names = []

    # create the init file
    with open(INIT_FILE, "w") as f:
        # add big comment saying this is an auto-generated file
        f.write("# This file is auto-generated by models/generate.py\n\n")

        # handle imports first
        f.write("import os\n")
        f.write("import sysconfig\n\n")

        # get the path to the blob folder
        f.write(f"_RELATIVE_BLOB_FOLDER = os.path.join('oakutils', 'blobs')\n")

        # get the site packages path
        f.write("_SITE_PACKAGES = sysconfig.get_paths()['purelib']\n")

        # get the path to the blob folder
        f.write(f"_BLOB_FOLDER = os.path.join(_SITE_PACKAGES, _RELATIVE_BLOB_FOLDER)\n")

        # add a space
        f.write("\n")

        # write the model names
        model_names = os.listdir(FINAL_BLOB_FOLDER)
        for model_name in model_names:
            if model_name == "__init__.py":
                continue
            var_name = model_name.upper().split(".")[0]
            # drop the first character since it is an underscore
            var_name = var_name[1:]
            var_names.append(var_name)
            f.write(f"{var_name} = os.path.abspath(os.path.join(_BLOB_FOLDER, '{model_name}'))\n")

        # write the __all__ variable
        f.write("\n__all__ = [\n")
        for var_name in var_names:
            f.write(f"    '{var_name}',\n")
        f.write("]\n")

if __name__ == "__main__":
    generate_onnx()
    generate_blobs()
    create_init()
