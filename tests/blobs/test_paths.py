# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from stdlib_list import stdlib_list
from oakutils.blobs.models import bulk


def test_all_paths_exists() -> None:
    """Test that all the paths exist."""
    stdlib = stdlib_list()
    for mp in dir(bulk):
        if mp[0] == "_":
            continue
        if mp in stdlib:
            continue
        model_paths = getattr(bulk, mp)
        if not isinstance(model_paths, tuple):
            continue
        for model_path in model_paths:
            assert model_path.exists()
            print(f" Found {mp}")
