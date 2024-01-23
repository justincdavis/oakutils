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

import os
from pathlib import Path

from oakutils.blobs import models


def parse_kernel_size(kernel_size: int) -> bool:
    """
    Use to parse a kernel size to ensure it is valid.

    Parameters
    ----------
    kernel_size : int
        The kernel size to parse
        Must be an odd integer
        Must be between 3 and 15 (since these are the compiled model sizes)

    Returns
    -------
    bool
        True if the kernel size is valid, False otherwise
    """
    valid = False
    min_kernel_size, max_kernel_size = 3, 15
    if (
        kernel_size % 2 == 0
        or kernel_size < min_kernel_size
        or kernel_size > max_kernel_size
    ):
        valid = False
    else:
        valid = True

    if not valid:
        err_msg = "Invalid kernel size, must be an odd integer between 3 and 15"
        raise ValueError(err_msg)
    return valid


def _valid_model_names(model_type: str) -> tuple[bool, list[str]]:
    """
    Use to check if a name is valid againist the names of compiled models.

    Parameters
    ----------
    model_type : str
        The model type to check

    Returns
    -------
    Tuple[bool, List[str]]
        A tuple of the validity of the model type and the list of valid names
    """
    valid_names = [
        "gaussian",
        "gftt",
        "gfttblur",
        "gfttgray",
        "gfttblurgray",
        "harris",
        "harrisblur",
        "harrisgray",
        "harrisblurgray",
        "hessian",
        "hessianblur",
        "hessiangray",
        "hessianblurgray",
        "laplacian",
        "gaussiangray",
        "laplaciangray",
        "laplacianblur",
        "laplacianblurgray",
        "sobel",
        "sobelgray",
        "sobelblur",
        "sobelblurgray",
        "pointcloud",
    ]
    valid_names.extend(
        [n.upper() for n in valid_names] + [n.capitalize() for n in valid_names],
    )
    for name in valid_names:
        if model_type in name:
            return True, valid_names
    return False, valid_names


def get_candidates(
    model_type: str,
    attributes: list[str],
    shaves: int,
) -> list[tuple[str, list[str], str]]:
    """
    Use to get the list of candidate models for a given model type and attribute.

    Parameters
    ----------
    model_type : str
        The model type to get candidates for.
        Examples of this are "gaussian", "laplacian", etc.
    attributes : List[str]
        The attribute to get candidates for.
        Examples of this are "kernel_size", etc.
    shaves : int
        The number of shaves to get candidates for.

    Returns
    -------
    List[Tuple[str, List[str], str]]
        The list of candidate models, each candidate is a tuple of the name, attributes, and path

    Raises
    ------
    ValueError
        If the model type is invalid (i.e. the name is not in the list of valid names)
    """
    valid, valid_names = _valid_model_names(model_type)
    if not valid:
        err_msg = f"Invalid model type, valid names are: {valid_names}"
        raise ValueError(err_msg)
    model_type = model_type.upper()

    potential_blobs = []
    model_module = getattr(models, f"shave{shaves}")
    # print(f"Checking module: {model_module}, for shaves: {shaves}")
    for model in [d for d in dir(model_module) if not d.startswith("_")]:
        # print(f"Looking for {model_type} in {model}")
        if model_type in model:
            blob_path = getattr(model_module, model)
            # print(f"Found {model_type} in {model}")
            # print(f"Blob path: {blob_path}")
            potential_blobs.append(blob_path)

    # parse the model names into 3 pieces, name, attribute, and extension
    candidate_blobs = []
    for blob in potential_blobs:
        path: str = Path(blob).name  # drop the extension
        path = os.path.split(path)[-1]  # just the file name
        data = path.split("_")  # split into name and attributes
        if len(data) == 1:  # if there are no extra attributes
            data = data[0].split(".")  # split on the dot to ensure good name
        name = data[0]  # name is the first piece
        # if the name is not equal to the model_type, maybe gaussian_gray instead of gaussian
        # print(f"Checking {name.upper()} against {model_type}")
        if model_type != name.upper():  # throw out if the case
            continue
        data.pop(0)  # remove name from list
        data = [d.split("X")[0] for d in data]  # split NxN attributes into N
        if "x" in data[0]:  # need to split on x or X
            data = [d.split("x")[0] for d in data]
        candidate_blobs.append((name, data, blob))  # add to list
    # print(f"Candidate blobs: {candidate_blobs}")

    # print(f"Checking attributes: {attributes}")

    candidate_models = []
    if len(attributes) == 0:  # if no attributes are given, return all models
        return candidate_blobs

    for name, attr_data, blob_path in candidate_blobs:  # for each model
        # print(f"Checking {attr_data} againist {attributes}")
        if attributes == attr_data:  # if the attribute is not in the model
            candidate_models.append((name, attr_data, blob_path))

    # print(f"Candidate models: {candidate_models}")
    return candidate_models
