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

import contextlib
import io
import logging

import blobconverter  # type: ignore[import]
import onnx
import onnxsim  # type: ignore[import]

_log = logging.getLogger(__name__)


def simplify(model_path: str, output_path: str, check_num: int = 5) -> None:
    """
    Simplifies a model using the onnxsim packages.

    Parameters
    ----------
    model_path : str
        The path to the model to simplify
    output_path : str
        The path to save the simplified model to
    check_num : int, optional
        The number of checks to perform on the simplified model, by default 5

    Raises
    ------
    AssertionError
        If the simplified model could not be validated
    """
    _log.debug("Simplifying model")
    model = onnx.load(model_path)
    with contextlib.redirect_stdout(io.StringIO()):
        model_simp, check = onnxsim.simplify(
            model,
            check_n=check_num,
            perform_optimization=True,
        )
    if not check:
        err_msg = "Simplified model could not be validated"
        raise AssertionError(err_msg)
    onnx.save(model_simp, output_path)


def compile_onnx(
    model_path: str,
    output_path: str,
    shaves: int = 6,
    version: str = "2022.1",
    *,
    simplify_model: bool | None = None,
) -> None:
    """
    Compiles an ONNX model to a blob saved at the output path.

    Parameters
    ----------
    model_path : str
        The path to the model to compile
    output_path : str
        The path to save the compiled model to
    shaves : int, optional
        The number of shaves to use, by default 6
    version : str, optional
        The version of blobconverter to use, by default "2022.1"
    simplify_model : bool, optional
        Whether or not to simplify the model before compiling, by default True

    Raises
    ------
    AssertionError
        If the simplified model could not be validated
    """
    if simplify_model is None:
        simplify_model = True

    if simplify_model:
        simplify(model_path, output_path)

    blobconverter.from_onnx(
        model=output_path,
        output_dir=output_path,
        data_type="FP16",
        use_cache=False,
        shaves=shaves,
        version=version,
        simplify=simplify_model,
    )
