from __future__ import annotations

import os

from oakutils.blobs import models


def parse_kernel_size(kernel_size: int) -> bool:
    """Parses a kernel size to ensure it is valid.

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
    if kernel_size % 2 == 0 or kernel_size < 3 or kernel_size > 15:
        valid = False
    else:
        valid = True

    if not valid:
        raise ValueError("Invalid kernel size, must be an odd integer between 3 and 15")
    return valid


def _valid_model_names(model_type: str) -> tuple[bool, list[str]]:
    """Checks if a name is valid againist the names of compiled models.

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
        "laplacian",
        "gaussiangray",
        "laplaciangray",
        "laplacianblur",
        "laplacianblurgray",
        "sobel",
        "sobelblur",
        "pointcloud",
    ]
    valid_names.extend(
        [n.upper() for n in valid_names] + [n.capitalize() for n in valid_names]
    )
    for name in valid_names:
        if model_type in name:
            return True, valid_names
    return False, valid_names


def get_candidates(
    model_type: str, attributes: list[str]
) -> list[tuple[str, list[str], str]]:
    """Gets the list of candidate models for a given model type and attribute.

    Parameters
    ----------
    model_type : str
        The model type to get candidates for.
        Examples of this are "gaussian", "laplacian", etc.
    attributes : List[str]
        The attribute to get candidates for.
        Examples of this are "kernel_size", etc.

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
        raise ValueError(f"Invalid model type, valid names are: {valid_names}")
    model_type = model_type.upper()

    potential_blobs = []
    for model in [d for d in dir(models) if not d.startswith("_")]:
        if model_type in model:
            blob_path = getattr(models, model)
            potential_blobs.append(blob_path.upper())

    # parse the model names into 3 pieces, name, attribute, and extension
    candidate_blobs = []
    for blob in potential_blobs:
        path: str = os.path.basename(blob)  # drop the extension
        path = os.path.split(path)[-1]  # just the file name
        data = path.split("_")  # split into name and attributes
        if len(data) == 1:  # if there are no extra attributes
            data = data[0].split(".")  # split on the dot to ensure good name
        name = data[0]  # name is the first piece
        # if the name is not equal to the model_type, maybe gaussian_gray instead of gaussian
        if model_type != name:  # throw out if the case
            continue
        data.pop(0)  # remove name from list
        data = [d.split("X")[0] for d in data]  # split NxN attributes into N
        candidate_blobs.append((name, data, blob))  # add to list

    candidate_models = []
    if len(attributes) == 0:  # if no attributes are given, return all models
        return candidate_blobs

    for attribute in attributes:  # for each attribute
        for name, attr_data, blob_path in candidate_blobs:  # for each model
            if attribute in attr_data:  # if the attribute is not in the model
                candidate_models.append((name, attr_data, blob_path))

    return candidate_models
