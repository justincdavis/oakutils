# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import functools
import itertools
import logging
import operator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from oakutils.blobs._analysis import LayerData, get_blob, get_layer_data
from oakutils.nodes import get_nn_data, get_nn_frame
from oakutils.vpu import VPU

if TYPE_CHECKING:
    import depthai as dai
    from typing_extensions import Self

_log = logging.getLogger(__name__)


class BlobEvaluater:
    """Evaluate blobs and their results."""

    def __init__(
        self: Self,
        blob_paths: list[Path | str],
    ) -> None:
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
        ValueError
            If all blobs do not have the same output shape

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
        output_shapes: set[tuple[tuple[int, ...], ...]] = set()
        for _, _, _, (input_layers, output_layers) in self._blobs:
            blob_input_shape: list[tuple[int, ...]] = [
                tuple(input_layer.shape) for input_layer in input_layers
            ]
            blob_output_shape: list[tuple[int, ...]] = [
                tuple(output_layer.shape) for output_layer in output_layers
            ]
            input_shapes.add(tuple(blob_input_shape))
            output_shapes.add(tuple(blob_output_shape))
        if len(input_shapes) > 1:
            err_msg = "All blobs must have the same input shape."
            err_msg += f" Found {len(input_shapes)} unique input shapes."
            err_msg += f" Input shapes: {input_shapes}"
            raise ValueError(err_msg)
        if len(output_shapes) > 1:
            err_msg = "All blobs must have the same output shape."
            err_msg += f" Found {len(output_shapes)} unique output shapes."
            err_msg += f" Output shapes: {output_shapes}"
            raise ValueError(err_msg)
        self._input_shape: tuple[int, ...] = input_shapes.pop()[0]
        self._output_shape: tuple[int, ...] = output_shapes.pop()[0]
        _log.debug(f"BlobEvaluator: Input shape: {self._input_shape}")
        _log.debug(f"BlobEvaluator: Output shape: {self._output_shape}")

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
        self._data: np.ndarray | list[np.ndarray] | None = None

    @property
    def results(
        self: Self,
    ) -> list[
        dai.ADatatype | list[dai.ADatatype] | list[dai.ADatatype | list[dai.ADatatype]]
    ]:
        """
        Get the last result of the execution batches.

        Returns an empty list of no results have been generated yet.

        Returns
        -------
        list[dai.ADatatype | list[dai.ADatatype] | list[dai.ADatatype | list[dai.ADatatype]]
            The results of each blob.

        """
        if self._results is None:
            return []
        return self._results

    @property
    def data(self: Self) -> np.ndarray | list[np.ndarray]:
        """
        Get the last data used to generate the results.

        Returns an empty list if no data has been used yet.

        Returns
        -------
        np.ndarray, list[np.ndarray]
            The data used to generate the results.

        """
        if self._data is None:
            return []
        return self._data

    def run(
        self: Self,
        data: np.ndarray | list[np.ndarray] | None = None,
        data_scale: float = 255.0,
    ) -> list[
        dai.ADatatype | list[dai.ADatatype] | list[dai.ADatatype | list[dai.ADatatype]]
    ]:
        """
        Run the models and get their results.

        Parameters
        ----------
        data : np.ndarray, list[np.ndarray] | None, optional
            The data to run through the models, by default None
            If None, then random data is used
        data_scale : float, optional
            The range (or maximum value) of the data, by default 255.0
            Represents a scaling factor for the random data which is intially
            generated as a [0, 1) range.

        Returns
        -------
        list[dai.ADatatype | list[dai.ADatatype] | list[dai.ADatatype | list[dai.ADatatype]]]
            The results of each blob.

        """
        results: list[
            dai.ADatatype
            | list[dai.ADatatype]
            | list[dai.ADatatype | list[dai.ADatatype]]
        ] = []
        rng = np.random.default_rng()
        if data is None:
            data = rng.random(self._input_shape).astype(np.float16) * data_scale
        self._data = data
        for idx, group in enumerate(self._allocations):
            group_blobs = [blob for _, blob, _, _ in group]
            eval_input = [data.copy() for _ in range(len(group))]
            _log.debug(
                f"BlobEvaluator: Running group {idx + 1} / {len(self._allocations)}",
            )
            with VPU() as vpu:
                vpu.reconfigure_multi(group_blobs)
                _log.debug(
                    f"BlobEvaluator: VPU reconfigured with {len(group_blobs)} blobs.",
                )
                batch_result = vpu.run(eval_input, safe=True)
                if not isinstance(batch_result, list):
                    batch_result = [batch_result]
                _log.debug(
                    f"BlobEvaluator: Batch {idx + 1} / {len(self._allocations)} completed.",
                )
                results.extend(batch_result)
        self._results = results
        return results

    def allclose(
        self: Self,
        data: list[np.ndarray | list[np.ndarray]] | None = None,
        rdiff: float = 1e-4,
        adiff: float = 1e-4,
        percentage: float = 99.0,
        *,
        image_output: bool | None = None,
        u8_input: bool | None = None,
    ) -> tuple[bool, list[tuple[int, int]]]:
        """
        Check if the results are all close to each other.

        Internally, this uses np.allclose.

        Parameters
        ----------
        data : list[np.ndarray | list[np.ndarray]] | None, optional
            The data to compare, by default None
            If None, then the last results are used
        rdiff : float, optional
            The relative tolerance, by default 1e-4
        adiff : float, optional
            The absolute tolerance, by default 1e-4
        percentage : float, optional
            The percentage of data which should be close, by default 1.0
            This will only be checked if the np.allclose call fails.
        image_output : bool, optional
            Whether the output is an image, by default None
            If None, will assume image outputs with shape
            (H, W, C, B) and perform data conversion accordingly.
            If output_shape does not have 4 dimensions, then
            will assume generic data reshape will work.
        u8_input : bool, optional
            Whether the input is uint8, by default None
            If None, will assume the input is float16

        Returns
        -------
        tuple[bool, list[tuple[int, int]]]
            The first element is True if all results are close, False otherwise.
            The second element is a list of pairs of indices that are not close.

        Raises
        ------
        ValueError
            If no data is provided and no results are available
        RuntimeError
            If automatic data conversion fails

        """
        if image_output is None:
            image_dims = 4
            image_output = len(self._output_shape) == image_dims

        # if data is None, use results and auto-convert
        if data is None:
            if self._results is None:
                err_msg = "No data provided and no results available."
                raise ValueError(err_msg)
            data = self._results
            # output shape is (W, H, C, B) for images
            # for other data types, will be different
            if image_output:
                channels = self._output_shape[2]
                frame_size: tuple[int, int] = self._output_shape[0:2]  # type: ignore[assignment]
                convert_func = functools.partial(
                    get_nn_frame,
                    channels=channels,
                    frame_size=frame_size,
                )
            else:
                convert_func = functools.partial(get_nn_data, use_first_layer=u8_input)
            try:
                # allow a try-except here since the conversion may fail
                # specifically if the models have multiple inputs/outputs
                converted_data = [convert_func(d) for d in data]  # type: ignore[arg-type]
            except (AttributeError, ValueError) as err:
                err_msg = f"Automatic data conversion failed for data: {data}"
                err_msg += (
                    " The issue may be caused by models with multiple inputs/outputs."
                )
                err_msg += " Please report this issue and attempt manual conversion."
                raise RuntimeError(err_msg) from err
        # if data has been provided, simply use that
        else:
            # ignore the type assignment here, since we are lowering from
            # a list[np.ndarray | list[np.ndarray]] to a list[np.ndarray]
            # which the subset is valid
            converted_data = data  # type: ignore[assignment]

        compare_data: list[tuple[tuple[int, np.ndarray], tuple[int, np.ndarray]]] = []
        for idx1, idx2 in itertools.combinations(range(len(converted_data)), 2):
            compare_data.append(
                (
                    (idx1, converted_data[idx1]),
                    (idx2, converted_data[idx2]),
                ),
            )

        non_matches = []
        for (idx1, d1), (idx2, d2) in compare_data:
            if not np.allclose(d1, d2, rtol=rdiff, atol=adiff):
                _log.debug(
                    f"Data {idx1} and {idx2} are not close via np.allclose checking %.",
                )
                # assess if a percentage of the results are close
                close_count = np.sum(np.isclose(d1, d2, rtol=rdiff, atol=adiff))
                close_percentage = close_count / np.prod(d1.shape)
                _log.debug(
                    f"Data {idx1} and {idx2} are {close_percentage * 100:.3f}% close.",
                )
                if close_percentage > (percentage / 100.0):
                    continue
                non_matches.append((idx1, idx2))
        return len(non_matches) == 0, non_matches
