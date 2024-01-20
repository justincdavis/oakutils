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
