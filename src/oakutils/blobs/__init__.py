"""
Module for compiling, running, and creating custom models.

Note:
----
    This module requires the [compiler] dependencies to be installed.
    If not only the `models` submodule will be available.

Submodules
----------
definitions
    Contains the definitions for the models.
models
    Contains the pre-compiled models.

Functions
---------
compile_model
    Compiles a model.
compile_onnx
    Compiles an onnx model into a blob.

"""
from . import models

__all__ = [
    "models",
]

try:
    from . import definitions
    from ._compiler import compile_model, compile_onnx

    __all__ = [
        "models",
        "definitions",
        "compile_model",
        "compile_onnx",
    ]
except ImportError:
    pass
