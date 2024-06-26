# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations


# for 3.8 compatibility
def remove_suffix(input_string: str, suffix: str) -> str:
    if suffix and input_string.endswith(suffix):
        return input_string[: -len(suffix)]
    return input_string


def dict_to_str(d: dict) -> str:
    """
    Use to convert a dictionary to a string by combining the values with underscores.

    Parameters
    ----------
    d : Dict
        The dictionary to convert

    Returns
    -------
    str
        The converted string

    """
    rv = "".join(
        [f"{v!s}x{v!s}_" if "kernel_size" in k else f"{v!s}_" for k, v in d.items()],
    )
    return remove_suffix(rv, "_")
