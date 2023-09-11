import itertools
import os
import shutil
import multiprocessing as mp

from oakutils.blobs import compile
from oakutils.blobs.definitions import AbstractModel, ModelType
from oakutils.blobs.definitions import (
    Gaussian,
    GaussianGray,
    Laplacian,
    LaplacianGray,
    LaplacianBlur,
    LaplacianBlurGray,
    Sobel,
    SobelBlur,
    SobelGray,
    SobelBlurGray,
    PointCloud,
)


def delete_folder(folder_path: str):
    if not os.path.exists(folder_path):
        return
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
    os.rmdir(folder_path)


def _compile_model(model_type, model_arg):
    print(f"Compiling {model_type.__name__} with args {model_arg}")
    model_path = compile(
        model_type,
        model_arg,
    )
    return model_path


def compile_model(model_type: AbstractModel):
    model_args = {}
    kernel_size_list = [3, 5, 7, 9, 11, 13, 15]
    arg_mapping = {
        ModelType.NONE: {},
        ModelType.KERNEL: {"kernel_size": kernel_size_list},
        ModelType.DUAL_KERNEL: {
            "kernel_size": kernel_size_list,
            "kernel_size2": kernel_size_list,
        },
    }
    if model_type.model_type() == ModelType.NONE:
        model_args = [{}]
    elif model_type.model_type() == ModelType.KERNEL:
        kernel_list = arg_mapping[model_type.model_type()]["kernel_size"]
        model_args = [{"kernel_size": t} for t in kernel_list]
    elif model_type.model_type() == ModelType.DUAL_KERNEL:
        kernel_list = arg_mapping[model_type.model_type()]["kernel_size"]
        kernel_list2 = arg_mapping[model_type.model_type()]["kernel_size2"]
        model_args = [
            {"kernel_size": t1, "kernel_size2": t2}
            for t1, t2 in itertools.product(
                kernel_list,
                kernel_list2,
            )
        ]
    else:
        raise RuntimeError("Unknown model type")

    # stolen from compiler code in oakutils internal
    def get_model_name(model_type, model_args):
        from oakutils.blobs._compiler.utils import dict_to_str

        arg_str = dict_to_str(model_args)

        # for 3.8 compatibility
        def remove_suffix(input_string, suffix):
            if suffix and input_string.endswith(suffix):
                return input_string[: -len(suffix)]
            return input_string

        # resolve the paths ahead of time for caching
        try:
            model_name = model_type.__name__
        except AttributeError:
            model_name = model_type.__class__.__name__

        model_name = remove_suffix(f"{model_name}_{arg_str}", "_")
        return model_name

    model_name = get_model_name(model_type, model_args[0])
    model_paths = [
        os.path.join(MODEL_FOLDER, f)
        for f in os.listdir(MODEL_FOLDER)
        if model_name == f.replace(".blob", "")
    ]
    if len(model_paths) != 0:
        print(f"Model {model_name} already exists, skipping...")
        return

    model_paths = []
    with mp.Pool() as pool:
        results = [
            pool.apply_async(_compile_model, args=(model_type, model_arg))
            for model_arg in model_args
        ]
        model_paths = [r.get() for r in results]

    for model_path in model_paths:
        shutil.copy(
            model_path, os.path.join(MODEL_FOLDER, os.path.basename(model_path))
        )


def compiles_models():
    models = [
        Gaussian,
        GaussianGray,
        Laplacian,
        LaplacianGray,
        LaplacianBlur,
        LaplacianBlurGray,
        Sobel,
        SobelBlur,
        SobelGray,
        SobelBlurGray,
        PointCloud,
    ]
    for model in models:
        try:
            compile_model(model)
        except Exception as e:
            print(f"Failed to compile model {model.__name__} with error {e}")

    # handle writing the __init__.py file
    var_names = []
    init_final_path = os.path.join(MODEL_FOLDER, "__init__.py")
    with open(init_final_path, "w") as f:
        # add big comment saying this is an auto-generated file
        f.write("# This file is auto-generated by scripts/compile_models.py\n\n")

        # handle imports first
        # f.write("import abc\n")
        f.write("import os\n")
        f.write("import site\n")
        f.write("import sysconfig\n\n")

        # get the path to the blob folder
        f.write(
            f"_RELATIVE_BLOB_FOLDER = os.path.join('oakutils', 'blobs', 'models')\n"
        )

        # get the site packages path
        f.write("_SITE_SITE_PACKAGES = site.getusersitepackages()\n")
        f.write("_SYSCONFIG_SITE_PACKAGES = sysconfig.get_paths()['purelib']\n")
        f.write(
            "_SITE_PACKAGES = _SITE_SITE_PACKAGES if os.name == 'posix' else _SYSCONFIG_SITE_PACKAGES\n"
        )

        # get the path to the blob folder
        f.write(f"_BLOB_FOLDER = os.path.join(_SITE_PACKAGES, _RELATIVE_BLOB_FOLDER)\n")

        # add a space
        f.write("\n")

        # # create the meta class definition
        # f.write("class _Blob(str):\n")
        # f.write("    __metaclass__ = abc.ABCMeta\n\n")
        # write the model names
        model_names = sorted(os.listdir(MODEL_FOLDER))
        for model_name in model_names:
            if model_name == "__init__.py":
                continue
            var_name = model_name.upper().split(".")[0]
            # # drop the first character since it is an underscore
            # var_name = var_name[1:]
            var_names.append(var_name)
            f.write(
                f"{var_name} = os.path.abspath(os.path.join(_BLOB_FOLDER, '{model_name}'))\n"
            )
            # f.write(f"{var_name} = _Blob(os.path.abspath(os.path.join(_BLOB_FOLDER, '{model_name}')))\n")
            # f.write(f"{var_name}.__doc__ = 'Absolute file path for {model_name} file'\n")

        # create the attributes section docstring in numpy format
        f.write("\n")
        f.write("'''\n")
        f.write("Note\n")
        f.write("-----\n")
        f.write("This module is auto-generated\n")
        f.write("Attributes\n")
        f.write("----------\n")
        for var_name in var_names:
            f.write(f"{var_name} : str\n")
            f.write(f"    Absolute file path for {var_name} file\n")
        f.write("'''\n")

        # write the __all__ variable
        f.write("\n__all__ = [\n")
        for var_name in var_names:
            f.write(f"    '{var_name}',\n")
        f.write("]\n")


def main():
    compiles_models()


if __name__ == "__main__":
    SCRIPT_PATH = os.path.realpath(os.path.dirname(__file__))
    MODEL_FOLDER = os.path.join(SCRIPT_PATH, "..", "src", "oakutils", "blobs", "models")
    main()
