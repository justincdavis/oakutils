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

Classes
-------
BenchmarkData
    Dataclass for storing benchmark data.
Metric
    Dataclass for storing metrics.

Functions
---------
bencmark_blob
    Benchmark a blob.
compile_model
    Compiles a model.
compile_onnx
    Compiles an onnx model into a blob.
clear_cache
    Clears the cache of compiled blobs.
get_blob
    Load a blob from a path.
get_cache_dir
    Get the cache directory path.
get_input_layer_data
    Get the input layer data for a blob.
get_output_layer_data
    Get the output layer data for a blob.
get_layer_data
    Get the input and output layer data for a blob.
get_model_name
    Get the name of a compiled model file.
get_model_path
    Get the path to the model blob.

"""

from __future__ import annotations

import logging

from . import models
from ._analysis import (
    get_blob,
    get_input_layer_data,
    get_layer_data,
    get_output_layer_data,
)
from ._benchmark import BenchmarkData, Metric, benchmark_blob
from ._find import get_model_path

_log = logging.getLogger(__name__)

__all__ = [
    "BenchmarkData",
    "Metric",
    "benchmark_blob",
    "get_blob",
    "get_input_layer_data",
    "get_layer_data",
    "get_model_path",
    "get_output_layer_data",
    "models",
]

_log.debug("Loaded blobs.models")

try:
    from . import definitions
    from ._compiler import (
        clear_cache,
        compile_model,
        compile_onnx,
        get_cache_dir,
        get_model_name,
    )

    __all__ += [
        "clear_cache",
        "compile_model",
        "compile_onnx",
        "definitions",
        "get_cache_dir",
        "get_model_name",
    ]

    _log.debug("Loaded blobs.definitions")
except ImportError:
    pass

_log.debug("Loaded blobs")
