# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import numpy as np

from oakutils import VPU
from oakutils.blobs.models.shave6 import GAUSSIAN_15X15, LAPLACIAN_15X15

from ..device import get_device_count


def test_reconfigure() -> None:
    if get_device_count() == 0:
        return

    rng = np.random.Generator(np.random.PCG64())
    with VPU() as vpu:
        vpu.reconfigure(GAUSSIAN_15X15)
        data = np.array(rng.integers(0, 255, (640, 480, 3)), dtype=np.uint8)
        vpu.run(data)


def test_multi_reconfigure() -> None:
    if get_device_count() == 0:
        return

    rng = np.random.Generator(np.random.PCG64())
    with VPU() as vpu:
        vpu.reconfigure_multi([GAUSSIAN_15X15, LAPLACIAN_15X15])
        data = np.array(rng.integers(0, 255, (640, 480, 3)), dtype=np.uint8)
        vpu.run([data, data])


def test_many_reconfigure() -> None:
    if get_device_count() == 0:
        return

    current = GAUSSIAN_15X15

    rng = np.random.Generator(np.random.PCG64())
    with VPU() as vpu:
        for _ in range(4):
            vpu.reconfigure(current)
            current = LAPLACIAN_15X15 if current == GAUSSIAN_15X15 else GAUSSIAN_15X15
            data = np.array(rng.integers(0, 255, (640, 480, 3)), dtype=np.uint8)
            vpu.run(data)


def test_many_multi_reconfigure() -> None:
    if get_device_count() == 0:
        return

    current = GAUSSIAN_15X15

    rng = np.random.Generator(np.random.PCG64())
    with VPU() as vpu:
        for _ in range(4):
            vpu.reconfigure_multi([current, current])
            current = LAPLACIAN_15X15 if current == GAUSSIAN_15X15 else GAUSSIAN_15X15
            data = np.array(rng.integers(0, 255, (640, 480, 3)), dtype=np.uint8)
            vpu.run([data, data])
