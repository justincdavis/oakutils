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
from . import models

__all__ = [
    "models",
]

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
except ImportError:
    pass
