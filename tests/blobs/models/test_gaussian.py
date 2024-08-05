# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from functools import partial

from oakutils.nodes import get_nn_frame
from oakutils.nodes.models import create_gaussian

from .load import create_model, run_model


def test_create():
    for ks in [3, 5, 7, 9, 11, 13, 15]:
        for shave in [1, 2, 3, 4, 5, 6]:
            for use_gs in [True, False]:
                modelfunc = partial(
                    create_gaussian,
                    kernel_size=ks,
                    shaves=shave,
                    grayscale_out=use_gs,
                )
                assert create_model(modelfunc) == 0, f"Failed for {ks}, {shave}, {use_gs}"
    return 0


def test_run():
    for ks in [3, 5, 7, 9, 11, 13, 15]:
        for shave in [1, 2, 3, 4, 5, 6]:
            for use_gs in [True, False]:
                modelfunc = partial(
                    create_gaussian,
                    kernel_size=ks,
                    shaves=shave,
                    grayscale_out=use_gs,
                )
                channels = 1 if use_gs else 3
                decodefunc = partial(
                    get_nn_frame,
                    channels=channels,
                )
                assert run_model(modelfunc, decodefunc) == 0, f"Failed for {ks}, {shave}, {use_gs}"
    return 0
