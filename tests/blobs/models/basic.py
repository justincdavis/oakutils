# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from collections.abc import Callable
from functools import partial

from oakutils.nodes import get_nn_frame

from .load import create_model, run_model


def create_model_ghhs(createmodelfunc: Callable):
    for use_blur in [True, False]:
        for ks in [3, 5, 7, 9, 11, 13, 15]:
            for shave in [1, 2, 3, 4, 5, 6]:
                for use_gs in [True, False]:
                    modelfunc = partial(
                        createmodelfunc,
                        blur_kernel_size=ks,
                        shaves=shave,
                        use_blur=use_blur,
                        grayscale_out=use_gs,
                    )
                    assert create_model(modelfunc) == 0, f"Failed for {ks}, {shave}, {use_blur}, {use_gs}"
    return 0


def run_model_ghhs(createmodelfunc: Callable):
    for use_blur in [True, False]:
        for ks in [3, 5, 7, 9, 11, 13, 15]:
            for shave in [1, 2, 3, 4, 5, 6]:
                for use_gs in [True, False]:
                    modelfunc = partial(
                        createmodelfunc,
                        blur_kernel_size=ks,
                        shaves=shave,
                        use_blur=use_blur,
                        grayscale_out=use_gs,
                    )
                    channels = 1 if use_gs else 3
                    decodefunc = partial(
                        get_nn_frame,
                        channels=channels,
                    )
                    assert run_model(modelfunc, decodefunc) == 0, f"Failed for {ks}, {shave}, {use_blur}, {use_gs}"
    return 0
