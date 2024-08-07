# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import copy
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
    from .hashs import get_bulk_tables, write_bulk_tables, hash_file, hash_bulk_entry, get_run_tables, write_model_tables
except ImportError:
    devicefile = Path(__file__).parent.parent.parent / "device.py"
    sys.path.append(str(devicefile.parent))
    from device import get_device_count
    from load import create_model, run_model
    from hashs import get_bulk_tables, write_bulk_tables, hash_file, hash_bulk_entry, get_run_tables, write_model_tables


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
    hash_table, run_table = get_run_tables()
    for use_blur in [True, False]:
        for ks in [3, 5, 7, 9, 11, 13, 15]:
            for shave in [1, 2, 3, 4, 5, 6]:
                for use_gs in [True, False]:
                    mname = copy.copy(modelname)
                    attributes = []
                    if use_blur:
                        mname += "blur"
                        attributes.append(str(ks))
                    if use_gs:
                        mname += "gray"
                    modelpath = get_model_path(mname, attributes, shave)
                    model_hash = hash_file(modelpath)
                    modelkey = modelpath.stem
                    # if the hash is the same and we have already gotten a successful run, continue
                    if hash_table[modelkey] == model_hash and run_table[modelkey]:
                        continue
                    # if the hash is not the same update the hash and set the run to false
                    existing_hash = hash_table[modelkey]
                    if existing_hash != model_hash:
                        hash_table[modelkey] = model_hash
                        run_table[modelkey] = False

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
                    retcode = run_model(modelfunc, decodefunc)
                    tableval = retcode == 0
                    run_table[modelkey] = tableval
                    write_model_tables(hash_table, run_table)
                    assert retcode == 0, f"Failed for {ks}, {shave}, {use_blur}, {use_gs}"


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


def check_model_equivalence(model_type: str, *, image_output: bool | None = None, u8_input: bool | None = None) -> None:
    models = get_models(model_type)
    hash_table, run_table = get_bulk_tables()
    for model_paths in models:
        if get_device_count() == 0:
            return
        modelkey = model_paths[0].stem[:-8]
        entryhash = hash_bulk_entry(model_paths)
        # if hash is the same and run_key is True, we can skip
        existinghash = hash_table[modelkey]
        if existinghash == entryhash and run_table[modelkey]:
            continue
        if existinghash != entryhash:
            hash_table[modelkey] = entryhash
            run_table[modelkey] = False
        # check if the model has already been run
        evaluator = BlobEvaluater([*model_paths])
        evaluator.run()
        success = evaluator.allclose(image_output=image_output, u8_input=u8_input)[0]
        run_table[modelkey] = success
        write_bulk_tables(hash_table, run_table)
        assert success, f"Failed allclose check for {model_paths}"
