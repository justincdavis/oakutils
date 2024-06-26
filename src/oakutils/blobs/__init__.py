# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
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
clear_cache
    Clears the cache of compiled blobs.


"""

from __future__ import annotations

import logging

from . import models

_log = logging.getLogger(__name__)

__all__ = [
    "models",
]

_log.debug("Loaded blobs.models")

try:
    from . import definitions
    from ._compiler import clear_cache, compile_model, compile_onnx

    __all__ = [
        "clear_cache",
        "compile_model",
        "compile_onnx",
        "definitions",
        "models",
    ]

    _log.debug("Loaded blobs.definitions")
except ImportError:
    pass

_log.debug("Loaded blobs")
