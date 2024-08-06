# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from oakutils.nodes.models import create_sobel

from .basic import create_model_ghhs, run_model_ghhs, check_model_equivalence


def test_create() -> None:
    create_model_ghhs(create_sobel)


def test_run() -> None:
    run_model_ghhs(create_sobel)


def test_equivalence() -> None:
    check_model_equivalence("sobel")
