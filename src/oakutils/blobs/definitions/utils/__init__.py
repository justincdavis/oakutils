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
