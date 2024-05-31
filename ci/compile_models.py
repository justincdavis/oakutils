# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
from __future__ import annotations

import argparse
import getpass
import itertools
import os
import shutil
import multiprocessing as mp
from io import TextIOWrapper
from pathlib import Path

from oakutils.blobs import compile_model as internal_compile_model
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
    Laserscan,
    Closing,
    ClosingBlur,
    ClosingGray,
    ClosingBlurGray,
    Dilation,
    DilationBlur,
    DilationGray,
    DilationBlurGray,
    Erosion,
    ErosionBlur,
    ErosionGray,
    ErosionBlurGray,
    Opening,
    OpeningBlur,
    OpeningGray,
    OpeningBlurGray,
    Harris,
    HarrisBlur,
    HarrisGray,
    HarrisBlurGray,
    Hessian,
    HessianBlur,
    HessianGray,
    HessianBlurGray,
    GFTT,
    GFTTBlur,
    GFTTGray,
    GFTTBlurGray,
)


def write_copyright(f: TextIOWrapper):
    f.write("# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)\n")
    f.write("# This program is free software: you can redistribute it and/or modify\n")
    f.write("# it under the terms of the GNU General Public License as published by\n")
    f.write("# the Free Software Foundation, either version 3 of the License, or\n")
    f.write("# (at your option) any later version.\n")
    f.write("#\n")
    f.write("# This program is distributed in the hope that it will be useful,\n")
    f.write("# but WITHOUT ANY WARRANTY; without even the implied warranty of\n")
    f.write("# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the\n")
    f.write("# GNU General Public License for more details.\n")
    f.write("#\n")
    f.write("# You should have received a copy of the GNU General Public License\n")
    f.write(
        "# along with this program. If not, see <https://www.gnu.org/licenses/>.\n\n"
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


def _compile_model(model_type, model_arg, shave):
    print(f"Compiling {model_type.__name__} with args {model_arg}, shaves {shave}")
    model_path = internal_compile_model(
        model_type,
        model_arg,
        shaves=shave,
        cache=False,  # don't cache the model (or load from cache) since we compile for all shaves
    )
    return model_path


def compile_model(model_type: AbstractModel, shave: int):
    model_args = {}
    kernel_size_list = [3, 5, 7, 9, 11, 13, 15]
    width_list = [5, 10, 20]
    scan_list = [1, 3, 5]
    arg_mapping = {
        ModelType.NONE: {},
        ModelType.KERNEL: {"kernel_size": kernel_size_list},
        ModelType.DUAL_KERNEL: {
            "kernel_size": kernel_size_list,
            "kernel_size2": kernel_size_list,
        },
        ModelType.LASERSCAN: {
            "width": width_list,
            "scans": scan_list,
        }
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
    elif model_type.model_type() == ModelType.LASERSCAN:
        width_list = arg_mapping[model_type.model_type()]["width"]
        scan_list = arg_mapping[model_type.model_type()]["scans"]
        model_args = [
            {"width": w, "scans": s}
            for w, s in itertools.product(
                width_list,
                scan_list,
            )
        ]
    else:
        raise RuntimeError("Unknown model type")

    # stolen from compiler code in oakutils internal
    def get_model_name(model_type, model_args):
        from oakutils.blobs._compiler.utils import dict_to_str

        arg_str = dict_to_str(model_args)

        # for 3.8 compatibility
        def remove_suffix(input_string: str, suffix: str):
            if suffix and input_string.endswith(suffix):
                return input_string[: -len(suffix)]
            return input_string

        # resolve the paths ahead of time for caching
        try:
            model_name = model_type.__name__
        except AttributeError:
            model_name = model_type.__class__.__name__

        model_name_arg = remove_suffix(f"{model_name}_{arg_str}", "_")
        return model_name_arg

    shave_folder = os.path.join(MODEL_FOLDER, f"shave{shave}")
    # create the shave folder if it doesn't exist
    if not os.path.exists(shave_folder):
        os.makedirs(shave_folder)

    missing_model_args = []
    for model_arg in model_args:
        model_name_arg = get_model_name(model_type, model_arg)
        model_paths = [
            os.path.join(shave_folder, f)
            for f in os.listdir(shave_folder)
            if model_name_arg == f.replace(".blob", "")
        ]
        if len(model_paths) == 0:
            missing_model_args.append(model_arg)
        else:
            print(
                f"Model {model_name_arg} with {shave} shaves already exists, skipping..."
            )
    model_args = missing_model_args

    try:
        model_paths = []
        with mp.Pool() as pool:
            results = [
                pool.apply_async(_compile_model, args=(model_type, model_arg, shave))
                for model_arg in model_args
            ]
            model_paths = [r.get() for r in results]
    except RuntimeError as e:
        # if a runtime error occurs, it could be because the blobconverter
        # cache got corrupted, so delete the cache and try again
        # 1: get cache directory and check if it exists
        username = getpass.getuser()
        linux_cache_dir = Path(f"/home/{username}/.cache/blobconverter")
        windows_cache_dir = Path(f"C:/Users/{username}/.cache/blobconverter")
        cache_dir = linux_cache_dir if os.name == "posix" else windows_cache_dir
        if not cache_dir.exists():
            raise e

        # 2: delete the cache directory
        delete_folder(str(cache_dir.resolve()))

        # 3: process the compilations in serial
        model_paths = []
        for model_arg in model_args:
            model_path = _compile_model(model_type, model_arg, shave)
            model_paths.append(model_path)

    # copy the models to the correct folders internally
    for model_path in model_paths:
        shutil.copy(
            model_path, os.path.join(shave_folder, os.path.basename(model_path))
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
        Laserscan,
        # Closing,
        # ClosingBlur,
        # ClosingGray,
        # ClosingBlurGray,
        # Dilation,
        # DilationBlur,
        # DilationGray,
        # DilationBlurGray,
        # Erosion,
        # ErosionBlur,
        # ErosionGray,
        # ErosionBlurGray,
        # Opening,
        # OpeningBlur,
        # OpeningGray,
        # OpeningBlurGray,
        Harris,
        HarrisBlur,
        HarrisGray,
        HarrisBlurGray,
        Hessian,
        HessianBlur,
        HessianGray,
        HessianBlurGray,
        GFTT,
        GFTTBlur,
        GFTTGray,
        GFTTBlurGray,
    ]
    shaves = [1, 2, 3, 4, 5, 6]

    # store failed models
    failed_models: list[tuple[AbstractModel, int, str]] = []

    # compile models and write the __init__.py files
    for shave in shaves:
        for model in models:
            try:
                compile_model(model, shave)
            except Exception as e:
                print(f"Failed to compile model {model.__name__} with error {e}")
                failed_models.append((model, shave, str(e)))

        # handle writing the __init__.py file
        var_names = []
        shave_model_folder = os.path.join(MODEL_FOLDER, f"shave{shave}")
        init_final_path = os.path.join(shave_model_folder, "__init__.py")
        with open(init_final_path, "w") as f:
            # add the copyright notice
            write_copyright(f)

            # add big comment saying this is an auto-generated file
            f.write(
                "# =============================================================================\n"
            )
            f.write("# This file is auto-generated by scripts/compile_models.py\n")
            f.write(
                "# =============================================================================\n\n"
            )

            # write the docstring
            f.write('"""\n')
            f.write(f"Module for {shave} shave models.\n\n")
            f.write("Note\n")
            f.write("----\n")
            f.write("This module is auto-generated\n\n")
            f.write("Attributes\n")
            f.write("----------\n")
            model_names = sorted(os.listdir(shave_model_folder))
            for model_name in model_names:
                if model_name == "__init__.py":
                    continue
                if model_name == "__pycache__":
                    continue
                model_name = model_name.split(".")[0]
                f.write(f"{model_name.capitalize()} : str\n")
                f.write(f"    nn.Module wrapper for {model_name.lower()} operation.\n")
            f.write('"""\n')
            f.write("from __future__ import annotations\n\n")

            # handle imports first
            # f.write("import abc\n")
            # f.write("import os\n")
            f.write("from pathlib import Path\n")
            f.write("import pkg_resources\n\n")

            # get the path to the blob folder
            f.write(
                f"_RELATIVE_BLOB_FOLDER: Path = Path('oakutils') / 'blobs' / 'models' / 'shave{shave}'\n"
            )

            # get the site packages path
            f.write(
                "_PACKAGE_LOCATION: str = pkg_resources.get_distribution('oakutils').location\n"
            )

            # perform a check on _PACKAGE_LOCATION since it could be None?
            f.write("if _PACKAGE_LOCATION is None:\n")
            f.write("    err_msg = 'Could not find package location'\n")
            f.write("    raise RuntimeError(err_msg)\n")

            # perform a check on _PACKAGE_LOCATION to ensure it exists and is a directory
            f.write("_PACKAGE_LOCATION_PATH: Path = Path(_PACKAGE_LOCATION)\n")
            f.write("if not _PACKAGE_LOCATION_PATH.exists():\n")
            f.write("    err_msg = 'Package location does not exist'\n")
            f.write("    raise RuntimeError(err_msg)\n")
            f.write("if not _PACKAGE_LOCATION_PATH.is_dir():\n")
            f.write("    err_msg = 'Package location is not a directory'\n")
            f.write("    raise RuntimeError(err_msg)\n")

            # get the path to the blob folder
            f.write(
                f"_BLOB_FOLDER: Path = Path(_PACKAGE_LOCATION) / _RELATIVE_BLOB_FOLDER\n"
            )

            # add a space
            f.write("\n")

            # # create the meta class definition
            # f.write("class _Blob(str):\n")
            # f.write("    __metaclass__ = abc.ABCMeta\n\n")
            # write the model names
            model_names = sorted(os.listdir(shave_model_folder))
            for model_name in model_names:
                if model_name == "__init__.py":
                    continue
                if model_name == "__pycache__":
                    continue
                var_name = model_name.upper().split(".")[0]
                # # drop the first character since it is an underscore
                # var_name = var_name[1:]
                var_names.append(var_name)
                f.write(
                    f"{var_name}: Path = Path(Path(_BLOB_FOLDER) / '{model_name}').resolve()\n"
                )
                # f.write(f"{var_name} = _Blob(os.path.abspath(os.path.join(_BLOB_FOLDER, '{model_name}')))\n")
                # f.write(f"{var_name}.__doc__ = 'Absolute file path for {model_name} file'\n")

            # # create the attributes section docstring in numpy format
            # f.write("\n")
            # f.write("'''\n")
            # f.write("Note\n")
            # f.write("-----\n")
            # f.write("This module is auto-generated\n")
            # f.write("Attributes\n")
            # f.write("----------\n")
            # for var_name in var_names:
            #     f.write(f"{var_name} : str\n")
            #     f.write(f"    Absolute file path for {var_name} file\n")
            # f.write("'''\n")

            # write the __all__ variable
            f.write("\n__all__ = [\n")
            for var_name in var_names:
                f.write(f"    '{var_name}',\n")
            f.write("]\n")

    with open(os.path.join(MODEL_FOLDER, "__init__.py"), "w") as f:
        write_copyright(f)
        # add big comment saying this is an auto-generated file
        f.write(
            "# =============================================================================\n"
        )
        f.write("# This file is auto-generated by scripts/compile_models.py\n")
        f.write(
            "# =============================================================================\n\n"
        )
        f.write('"""\n')
        f.write("Module for compiled models.\n\n")
        f.write("Note\n")
        f.write("-----\n")
        f.write("This module is auto-generated\n\n")
        f.write("Attributes\n")
        f.write("----------\n")
        for idx, shave in enumerate(shaves):
            f.write(f"shave{shave} : module\n")
            f.write(f"    Contains all the models compiled for {shaves[idx]} shaves\n")
        f.write("\n")
        f.write('"""\n')
        f.write("from __future__ import annotations\n\n")
        f.write("import importlib\n")
        f.write("import sys\n")
        f.write("from typing import TYPE_CHECKING\n\n")

        # not used due to lazy imports
        # # from . import shave modules
        # for shave in shaves:
        #     f.write(f"from . import shave{shave}\n")
        # f.write("\n")

        # add TYPE_CHECKING import
        f.write("if TYPE_CHECKING:\n")
        f.write("    from types import ModuleType\n\n")

        # create _submodules for lazy loading
        f.write("_submodules = [\n")
        for shave in shaves:
            f.write(f"    'shave{shave}',\n")
        f.write("]\n\n")

        # create _loaded_modules for lazy loading
        f.write("_loaded_modules: dict[str, ModuleType | None] = {\n")
        for shave in shaves:
            f.write(f'    "shave{shave}": None,\n')
        f.write("}\n\n")

        # write the new __getattr__ function
        f.write("def __getattr__(name: str) -> ModuleType:\n")
        f.write("    if name in _submodules:\n")
        f.write('        _loaded_modules[name] = importlib.import_module(f"{__name__}.{name}")\n')
        f.write("        setattr(sys.modules[__name__], name, _loaded_modules[name])\n")
        f.write("        module = _loaded_modules[name]\n")
        f.write("        if module is None:\n")
        f.write('            err_msg = f"Could not import module {name}"\n')
        f.write("            raise ImportError(err_msg)\n")
        f.write("        return module\n")
        f.write('    err_msg = f"module {__name__} has no attribute {name}"\n')
        f.write("    raise AttributeError(err_msg)\n\n")

        # write new __dir__ method
        f.write("def __dir__() -> list[str]:\n")
        f.write("    module_attrs = list(object.__dir__(sys.modules[__name__]))\n")
        f.write("    return list(set(module_attrs + _submodules))\n\n")

        # write the __all__ variable
        f.write("\n__all__ = [\n")
        for shave in shaves:
            f.write(f"    'shave{shave}',\n")
        f.write("]\n")

    # write a summary of the failed models
    if len(failed_models) > 0:
        print(f"\nFailed to compile the following {len(failed_models)} models:")
        print("===============================================================")
        for model, shave, error in failed_models:
            print(f"Model: {model.__name__}, with {shave} shaves.\n\tFailed with error: {error}")

def verify_blobs():
    shaves = [1, 2, 3, 4, 5, 6]
    # verify that each shave folder has the same number of models
    num_files = []
    for shave in shaves:
        shave_model_folder = os.path.join(MODEL_FOLDER, f"shave{shave}")
        shave_folder_files = os.listdir(shave_model_folder)
        valid_files = [f for f  in shave_folder_files if not f.startswith("_")]
        num_files.append(len(valid_files))
    if len(set(num_files)) != 1:
        for idx, nf in enumerate(num_files):
            print(f"Shave {idx + 1}: {nf} files")

        raise RuntimeError("Not all shave folders have the same number of files")

    # verify each __init__.py file has the same number of models
    num_files = []
    for shave in shaves:
        shave_model_folder = os.path.join(MODEL_FOLDER, f"shave{shave}")
        init_final_path = os.path.join(shave_model_folder, "__init__.py")
        with open(init_final_path) as f:
            num_files.append(len(f.readlines()))
    if len(set(num_files)) != 1:
        for idx, nf in enumerate(num_files):
            print(f"Shave {idx + 1}: {nf} lines")

        raise RuntimeError("Not all __init__.py files have the same number of lines")


def build_from_cache():
    from oakutils.blobs._compiler.paths import get_cache_dir_path

    cache_dir = get_cache_dir_path()
    cache_dir = os.path.join(cache_dir, "blobs")
    for modeltype in os.listdir(cache_dir):
        modeltype_dir = os.path.join(cache_dir, modeltype)
        for model in os.listdir(modeltype_dir):
            data = model.split(".")  # removes the .blob
            modelname, shaveinfo = data[0], data[1]
            data = shaveinfo.split("_")  # splits into name and attributes
            shaves = int(data[-1].replace("shave", ""))
            modelname = modelname.split("_")[:-3]
            strname = ""
            for name in modelname:
                strname += name + "_"
            strname = strname[:-1]
            shutil.copy(
                os.path.join(modeltype_dir, model),
                os.path.join(MODEL_FOLDER, f"shave{shaves}", strname + ".blob"),
            )


def main():
    # copies compiled models from oakutils cache directory
    # and into correct folder structure
    if BUILD_FROM_CACHE:
        build_from_cache()
        verify_blobs()
    # rebuils all models from definition files, will skip
    # if model already exists
    if BUILD_FROM_DEFINITIONS:
        compiles_models()
        verify_blobs()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Builds the models from the cache",
    )
    parser.add_argument(
        "--definitions",
        action="store_true",
        help="Builds the models from the definitions",
    )
    args = parser.parse_args()

    BUILD_FROM_CACHE = args.cache
    BUILD_FROM_DEFINITIONS = args.definitions
    if not BUILD_FROM_CACHE and not BUILD_FROM_DEFINITIONS:
        raise RuntimeError("Must specify either --cache or --definitions")

    SCRIPT_PATH = os.path.realpath(os.path.dirname(__file__))
    MODEL_FOLDER = os.path.join(SCRIPT_PATH, "..", "src", "oakutils", "blobs", "models")
    # create the model folder if it doesn't exist
    if not os.path.exists(MODEL_FOLDER):
        os.makedirs(MODEL_FOLDER)
    main()
