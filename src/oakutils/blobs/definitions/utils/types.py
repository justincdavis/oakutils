# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Module which defines types for use in the blob definitions.

Classes
-------
ModelType
    Represents the different arguments a model constructor can take.
InputType
    Represents the type of a given input to a model in the forward call
    E.g. FP16, U8, etc.

Functions
---------
input_type_to_str
    Convert an InputType to a string.
"""

from __future__ import annotations

from enum import Enum


class ModelType(Enum):
    """Represents the different arguments a model constructor can take."""

    NONE = 0
    KERNEL = 1
    DUAL_KERNEL = 2
    LASERSCAN = 3


class InputType(Enum):
    """Represents the type of a given input to a model in the forward call."""

    FP16 = 0
    U8 = 1
    XYZ = 2


def input_type_to_str(inputtype: InputType) -> str:
    """
    Convert an InputType to a string.

    Parameters
    ----------
    inputtype : InputType
        The input type to convert

    Returns
    -------
    str
        The converted string

    Raises
    ------
    ValueError
        If the input type is unknown

    """
    if inputtype == InputType.FP16:
        return "FP16"
    if inputtype == InputType.U8:
        return "U8"
    if inputtype == InputType.XYZ:
        return "FP16"
    err_msg = f"Unknown input type: {inputtype}"
    raise ValueError(err_msg)
