# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

from stdlib_list import stdlib_list
from oakutils.nodes import get_nn_frame
from oakutils.blobs import get_model_path
from oakutils.blobs.models import bulk
from oakutils.blobs.testing import BlobEvaluater

try:
    from ...device import get_device_count
    from .load import create_model, run_model
except ImportError:
    devicefile = Path(__file__).parent.parent.parent / "device.py"
    sys.path.append(str(devicefile.parent))
    from device import get_device_count

    from load import create_model, run_model


def create_model_ghhs(createmodelfunc: Callable) -> None:
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
                    assert (
                        create_model(modelfunc) == 0
                    ), f"Failed for {ks}, {shave}, {use_blur}, {use_gs}"


def run_model_ghhs(createmodelfunc: Callable, modelname: str) -> None:
    for use_blur in [True, False]:
        for ks in [3, 5, 7, 9, 11, 13, 15]:
            for shave in [1, 2, 3, 4, 5, 6]:
                for use_gs in [True, False]:
                    # check if the model
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
                    assert (
                        run_model(modelfunc, decodefunc) == 0
                    ), f"Failed for {ks}, {shave}, {use_blur}, {use_gs}"


def get_models(model_type: str) -> list[tuple[Path, ...]]:
    stdlib = stdlib_list()
    models = []
    for mp in dir(bulk):
        mt = model_type.upper()
        if mp[0] == "_":
            continue
        if mp in stdlib:
            continue
        if mt not in mp:
            continue
        model_paths = getattr(bulk, mp)
        if not isinstance(model_paths, tuple):
            continue
        # if we found a tuple of paths, add it to the list
        models.append(model_paths)
    return models


def check_model_equivalence(model_type: str) -> None:
    models = get_models(model_type)
    for model_paths in models:
        if get_device_count() == 0:
            return
        evaluator = BlobEvaluater([*model_paths])
        evaluator.run()
        assert evaluator.allclose()[0], f"Failed allclose check for {model_paths}"
