# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import logging
import operator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from oakutils.blobs._analysis import LayerData, get_blob, get_layer_data
from oakutils.vpu import VPU

if TYPE_CHECKING:
    import depthai as dai
    from typing_extensions import Self

_log = logging.getLogger(__name__)


class BlobEvaluater:
    """Evaluate blobs and their results."""

    def __init__(self: Self, blob_paths: list[Path | str]) -> None:
        """
        Initialize the BlobEvaluater.

        Parameters
        ----------
        blob_paths : list[Path | str]
            The paths to the blobs to evaluate.
            All blobs must have the same input and outputs shapes.
            It is assumed that all blobs perform the same task.

        Raises
        ------
        FileNotFoundError
            If a blob does not exist.
        ValueError
            If all blobs do not have the same input shape.

        """
        self._blob_lookup: dict[
            int,
            tuple[int, Path, int, tuple[list[LayerData], list[LayerData]]],
        ] = {}
        for idx, bp in enumerate(blob_paths):
            blob_path = Path(bp)
            if not blob_path.exists():
                err_msg = f"Blob {blob_path} does not exist."
                raise FileNotFoundError(err_msg)
            vino_blob = get_blob(blob_path)
            shaves = vino_blob.numShaves
            layer_data = get_layer_data(vino_blob)
            self._blob_lookup[idx] = (idx, blob_path, shaves, layer_data)
        self._blobs = sorted(
            self._blob_lookup.values(),
            key=operator.itemgetter(2),
        )

        # verify that all blobs have the same input shape
        input_shapes: set[tuple[tuple[int, ...], ...]] = set()
        for _, _, _, (input_layers, _) in self._blobs:
            blob_input_shape: list[tuple[int, ...]] = [
                tuple(input_layer.shape) for input_layer in input_layers
            ]
            input_shapes.add(tuple(blob_input_shape))
        if len(input_shapes) > 1:
            err_msg = "All blobs must have the same input shape."
            err_msg += f" Found {len(input_shapes)} unique input shapes."
            err_msg += f" Input shapes: {input_shapes}"
            raise ValueError(err_msg)
        self._input_shape: tuple[int, ...] = input_shapes.pop()[0]
        _log.debug(f"Input shape: {self._input_shape}")

        # setup the allocations (groups with shaves <= 12)
        max_shaves = 12
        self._allocations: list[
            list[
                tuple[
                    int,
                    Path,
                    int,
                    tuple[list[LayerData], list[LayerData]],
                ]
            ]
        ] = []
        current_shaves = 0
        current_group: list[
            tuple[int, Path, int, tuple[list[LayerData], list[LayerData]]]
        ] = []
        for idx, blob, shaves, layer_info in self._blobs:
            if current_shaves + shaves > max_shaves:
                _log.debug(
                    f"New group created, shaves: {current_shaves}, models: {len(current_group)}.",
                )
                self._allocations.append(current_group)
                current_group = []
                current_shaves = 0
            current_group.append((idx, blob, shaves, layer_info))
            current_shaves += shaves
        if current_group:
            _log.debug(
                f"New group created, shaves: {current_shaves}, models: {len(current_group)}.",
            )
            self._allocations.append(current_group)
        _log.debug(f"Number allocation groups: {len(self._allocations)}")

        # allocate a spot for the results
        self._results: list | None = None

    def run(
        self: Self,
        data: list[np.ndarray | list[np.ndarray]] | None = None,
    ) -> list[
        dai.ADatatype | list[dai.ADatatype] | list[dai.ADatatype | list[dai.ADatatype]]
    ]:
        """
        Run the models and get their results.

        Parameters
        ----------
        data : list[np.ndarray, list[np.ndarray]] | None, optional
            The data to run through the models, by default None
            If None, then random data is used

        Returns
        -------
        list[dai.ADatatype | list[dai.ADatatype] | list[dai.ADatatype | list[dai.ADatatype]]]
            The results of each blob.

        """
        results = []
        rng = np.random.default_rng()
        for idx, group in enumerate(self._allocations):
            if data is not None:
                eval_input = data.copy()
            else:
                eval_input = [
                    rng.random(self._input_shape).astype(np.float32)
                    for _ in range(len(self._blobs))
                ]
            group_blobs = [blob for _, blob, _, _ in group]
            _log.debug(
                f"BlobEvaluator: Running group {idx + 1} / {len(self._allocations)}",
            )
            with VPU() as vpu:
                vpu.reconfigure_multi(group_blobs)
                _log.debug(
                    f"BlobEvaluator: VPU reconfigured with {len(group_blobs)} blobs.",
                )
                batch_result = vpu.run(eval_input, safe=True)
                _log.debug(
                    f"BlobEvaluator: Batch {idx + 1} / {len(self._allocations)} completed.",
                )
                results.append(batch_result)
        self._results = results
        return results
