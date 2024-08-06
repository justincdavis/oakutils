# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from functools import partial

from oakutils.nodes.models import create_laserscan, get_laserscan

from .basic import check_model_equivalence
from .load import create_model, run_model


def test_create() -> None:
    for width in [5, 10, 20]:
        for scan in [1, 3, 5]:
            for shave in [1, 2, 3, 4, 5, 6]:
                modelfunc = partial(
                    create_laserscan,
                    width=width,
                    scans=scan,
                    shaves=shave,
                )
                assert create_model(modelfunc) == 0, f"Failed for {width}, {scan}, {shave}"


def test_run() -> None:
    for width in [5, 10, 20]:
        for scan in [1, 3, 5]:
            for shave in [1, 2, 3, 4, 5, 6]:
                modelfunc = partial(
                    create_laserscan,
                    width=width,
                    scans=scan,
                    shaves=shave,
                )
                decodefunc = get_laserscan
                assert run_model(modelfunc, decodefunc) == 0, f"Failed for {width}, {scan}, {shave}"


def test_equivalence() -> None:
    check_model_equivalence("laserscan")
