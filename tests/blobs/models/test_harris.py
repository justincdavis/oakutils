# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from oakutils.nodes.models import create_harris

from .basic import create_model_ghhs, run_model_ghhs


def test_create() -> int:
    return create_model_ghhs(create_harris)


def test_run():
    return run_model_ghhs(create_harris)
