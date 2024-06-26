# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Module for classes and functions for use in defining blobs.

Classes
-------
InputType
    Represents the type of a given input to a model in the forward call
    E.g. FP16, U8, etc.
ModelType
    Represents the different arguments a model constructor can take.

Functions
---------
convert_to_fp16
    Convert a U8 input to FP16.
input_type_to_str
    Convert an InputType to a string.
"""

from .conversion import convert_to_fp16
from .types import InputType, ModelType, input_type_to_str

__all__ = ["InputType", "ModelType", "convert_to_fp16", "input_type_to_str"]
