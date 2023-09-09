from __future__ import annotations

from enum import Enum


class ModelType(Enum):
    """Represents the different arguments a model constructor can take."""

    NONE = 0
    KERNEL = 1
    DUAL_KERNEL = 2


class InputType(Enum):
    """Represents the type of a given input to a model in the forward call
    E.g. FP16, U8, etc.
    """

    FP16 = 0
    U8 = 1
    XYZ = 2


def input_type_to_str(inputtype: InputType) -> str:
    """Convert an InputType to a string.

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
    raise ValueError(f"Unknown input type: {inputtype}")
