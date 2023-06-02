from typing import List, Tuple

from ...blobs import models


def parse_kernel_size(kernel_size: int) -> bool:
    """
    Parses a kernel size to ensure it is valid

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
    if kernel_size % 2 == 0:
        valid = False
    elif kernel_size < 3 or kernel_size > 15:
        valid = False
    else:
        valid = True

    if not valid:
        raise ValueError("Invalid kernel size, must be an odd integer between 3 and 15")
    return valid


def _valid_model_names(model_type: str) -> Tuple[bool, List[str]]:
    valid_names = [
        "gaussian",
        "laplacian",
        "gaussiangray",
        "laplaciangray",
        "laplacianblur",
        "laplacianblurgray",
        "sobel",
        "sobelblur",
    ]
    for name in valid_names:
        if model_type in name:
            return True, valid_names
    return False, valid_names


def get_candidates(model_type: str, attributes: List[str]) -> List[str]:
    """
    Gets the list of candidate models for a given model type and attribute

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
    List[str]
        The list of candidate models

    Raises
    ------
    ValueError
        If the model type is invalid (i.e. the name is not in the list of valid names)
    """
    valid, valid_names = _valid_model_names(model_type)
    if not valid:
        raise ValueError(f"Invalid model type, valid names are: {valid_names}")

    potential_blobs = []
    for model in [d for d in dir(models) if not d.startswith("_")]:
        if model_type in model:
            blob_path = getattr(models, model)
            potential_blobs.append(blob_path)

    # parse the model names into 3 pieces, name, attribute, and extension
    candidate_blobs = []
    for blob in potential_blobs:
        path: str = blob[:-5]  # remove .blob
        data = path.split("_")  # split into name and attributes
        name = data[0]  # name is the first piece
        # if the name is not equal to the model_type, maybe gaussian_gray instead of gaussian
        if model_type != name:  # throw out if the case
            continue
        data.pop(0)  # remove name from list
        data = [d.split("x")[0] for d in data]  # split NxN attributes into N
        candidate_blobs.append((name, data, blob))  # add to list

    candidate_models = candidate_blobs  # copy list
    for attribute in attributes:  # for each attribute
        for name, attr_data, blob_path in candidate_models:  # for each model
            if attribute not in attr_data:  # if the attribute is not in the model
                candidate_models.remove((name, attr_data, blob_path))

    return candidate_models
