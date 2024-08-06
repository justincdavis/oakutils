# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path

from oakutils.nodes.models._parsing import get_candidates as _get_candidates


def get_model_path(
    model_type: str,
    model_attributes: list[str],
    shaves: int,
) -> Path:
    """
    Get the path to the model blob.

    Parameters
    ----------
    model_type : str
        The model type to get the path for.
        Examples include: ['gaussian', 'sobel']
    model_attributes : list[str]
        The model attributes to get the path for.
        An example could be ['15'] for a gaussian model
        using a 15x15 kernel size.
    shaves : int
        The number of shaves the model was compiled for.

    Returns
    -------
    Path
        The path to the model blob.

    Raises
    ------
    FileNotFoundError
        If the returned model blob does not exists.
    ValueError
        If no model blob paths could be formed from the attributes and shaves.

    """
    candidates = _get_candidates(model_type, model_attributes, shaves)
    if len(candidates) == 0:
        err_msg = f"No model blob paths could be formed from the attributes {model_attributes} and shaves {shaves}."
        raise ValueError(err_msg)
    _, _, path = candidates[0]
    blobpath = Path(path)
    if not blobpath.exists():
        err_msg = f"The model blob path {blobpath} does not exists."
        raise FileNotFoundError(err_msg)
    return blobpath
