# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from oakutils.nodes import get_yolo_data
from oakutils.vpu import VPU

from ._analysis import get_input_layer_data

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Metric:
    mean: float
    std: float
    min: float
    max: float


@dataclass(frozen=True)
class BenchmarkData:
    latency: Metric


def benchmark_blob(
    blob: Path | str,
    iterations: int = 100,
    warmup_iterations: int = 10,
    *,
    is_yolo: bool | None = None,
    warmup: bool | None = None,
) -> BenchmarkData:
    """
    Benchmark a blob.

    Parameters
    ----------
    blob : Path | str
        The path to the blob file or directory containing the blob files.
    iterations : int, optional
        The number of iterations to run, by default 100
    warmup_iterations : int, optional
        The number of warmup iterations to run, by default 10
    is_yolo : bool, optional
        Whether the blob is a Yolo blob, by default None.
        If is_yolo is True, then the blob_path should be a directory
        and the directory should contain a .json file generated
        during compilation of the blob.
    warmup : bool, optional
        Whether to run warmup iterations, by default None.
        If warmup is True, then the warmup_iterations will be run
        before the iterations.

    Returns
    -------
    BenchmarkData
        The benchmark data.

    Raises
    ------
    FileNotFoundError
        If the blob file or directory does not exist.
    ValueError
        If is_yolo is True and the blob_path is not a directory.
    ValueError
        If the directory does not contain exactly one blob file.
    ValueError
        If the directory does not contain exactly one json file.

    """
    blob_path = Path(blob)
    if not blob_path.exists():
        err_msg = f"Blob or blob directory does not exists: {blob_path}"
        raise FileNotFoundError(err_msg)

    # create the vpu
    vpu = VPU()

    # two cases for blob
    if is_yolo:
        # if yolo data then blob_path should be a directory
        if not blob_path.is_dir():
            err_msg = "If is_yolo is True, then blob_path should be a directory."
            raise ValueError(err_msg)
        blob_files = list(blob_path.glob("*.blob"))
        _log.debug(f"Blob files: {blob_files}")
        if len(blob_files) != 1:
            err_msg = (
                f"Directory should contain only one blob file, found: {len(blob_files)}"
            )
            raise ValueError(err_msg)
        found_blob = blob_files[0]
        json_files = list(blob_path.glob("*.json"))
        _log.debug(f"Json files: {json_files}")
        if len(json_files) != 1:
            err_msg = (
                f"Directory should contain only one json file, found: {len(json_files)}"
            )
            raise ValueError(err_msg)
        json_path = json_files[0]
        yolo_data = get_yolo_data(json_path)
        vpu.reconfigure(found_blob, model_data=yolo_data)
        # set the blob_path variable to the found file
        blob_path = found_blob
    else:
        vpu.reconfigure(blob_path)

    # get input size of blob
    input_layer_data = get_input_layer_data(blob_path)
    input_shapes = [tuple(layer.shape) for layer in input_layer_data]

    rng = np.random.Generator(np.random.PCG64())
    if warmup:
        for _ in range(warmup_iterations):
            rand_input = [
                rng.random(shape).astype(np.float32) for shape in input_shapes
            ]
            vpu.run(rand_input)

    # run the actual benchmarking
    timings = []
    for _ in range(iterations):
        rand_input = [rng.random(shape).astype(np.float32) for shape in input_shapes]
        start = time.perf_counter()
        vpu.run(rand_input)
        end = time.perf_counter()
        timings.append(end - start)

    # calculate the metrics
    latency_mean = float(np.mean(timings))
    latency_std = float(np.std(timings))
    latency_min = float(np.min(timings))
    latency_max = float(np.max(timings))

    return BenchmarkData(
        latency=Metric(latency_mean, latency_std, latency_min, latency_max),
    )
