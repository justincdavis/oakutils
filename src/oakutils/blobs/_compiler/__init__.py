# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Module for compiling models and onnx files into blob files.

Functions
---------
compile_model
    Use to compile a model into a blob file.
compile_onnx
    Use to compile an onnx file into a blob file.
clear_cache
    Use to clear the cache of compiled blob files.
"""

from __future__ import annotations

from .compile import compile_model
from .onnx import compile_onnx
from .paths import clear_cache

__all__ = [
    "clear_cache",
    "compile_model",
    "compile_onnx",
]
