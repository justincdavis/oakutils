# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from types import ModuleType
from pathlib import Path
from functools import lru_cache

import numpy as np
import depthai as dai
from oakutils.vpu import VPU
from oakutils.blobs.models import shave1, shave2, shave3, shave4, shave5, shave6


class DataGen:
    def __init__(self, size: tuple[int, int, int], dtype: np.dtype = np.uint8) -> None:
        self.size = size
        self.dtype = dtype
        self._rng = np.random.Generator(np.random.PCG64())

    def __call__(self) -> np.ndarray:
        return self._rng.integers(0, 256, size=self.size, dtype=self.dtype)


@lru_cache
def get_models(shave: ModuleType) -> list[str]:
    return sorted([str(f) for f in dir(shave) if not f.startswith("_") and isinstance(f, Path)])

def get_all_models() -> list[tuple[str, str, str, str, str, str]]:
    shave1_models = get_models(shave1)
    shave2_models = get_models(shave2)
    shave3_models = get_models(shave3)
    shave4_models = get_models(shave4)
    shave5_models = get_models(shave5)
    shave6_models = get_models(shave6)

    return list(zip(shave1_models, shave2_models, shave3_models, shave4_models, shave5_models, shave6_models))

def get_models_keyword(models: list[tuple[str, str, str, str, str, str]], keyword: str) -> list[tuple[str, str, str, str, str, str]]:
    return [model for model in models if keyword in model]

def get_all_models_keyword(keyword: str) -> list[tuple[str, str, str, str, str, str]]:
    return get_models_keyword(get_all_models(), keyword)

def get_batch(models: list[tuple[str, str, str, str, str, str]], idx: int) -> tuple[tuple[str, str, str, str], tuple[str, str]]:
    return models[idx][:4], models[idx][4:]

def eval_model(model_type: str, input_size: tuple[int, int, int]):
    models = get_all_models_keyword(model_type)

    if len(dai.Device.getAllAvailableDevices()) == 0:
        return 0  # no device found

    vpu = VPU()
    gen = DataGen(input_size)

    for idx, (model_packet) in enumerate(models):
        batch1, batch2 = get_batch(model_packet)

        # load each batch onto device with VPU
        vpu.reconfigure_multi(list(batch1))
        data1 = [gen() for _ in range(len(batch1))]
        results1 = vpu.run(data1)

        vpu.reconfigure_multi(list(batch2))
        data2 = [gen() for _ in range(len(batch2))]
        results2 = vpu.run(data2)

        # compare results
        results = results1 + results2
        assert len(results) == len(model_packet)
        for res in results:
            if not np.allclose(res, results[0]):
                err_msg = f"Results are not the same id: {idx}, for models:\n"
                for m in model_packet:
                    err_msg += f"\t{m}\n"
                raise ValueError(err_msg)
        else:
            # if all results the same, hash the files and record
            pass

    vpu.stop()
