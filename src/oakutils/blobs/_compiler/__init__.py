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
