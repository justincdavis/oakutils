# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: TID252
"""Module for using the onboard VPU as a standalone processor."""

from __future__ import annotations

import atexit
import logging
from pathlib import Path
from queue import Empty, Queue
from threading import Condition, Thread
from typing import TYPE_CHECKING

import depthai as dai
import numpy as np

from oakutils.core import create_device
from oakutils.nodes import (
    MobilenetData,
    YolomodelData,
    create_mobilenet_detection_network,
    create_neural_network,
    create_xin,
    create_xout,
    create_yolo_detection_network,
)
from oakutils.nodes.buffer import MultiBuffer

if TYPE_CHECKING:
    from typing_extensions import Self

_log = logging.getLogger(__name__)


class VPU:
    """Class for using the onboard VPU as a standalone processor."""

    def __init__(
        self: Self,
        device_id: str | None = None,
    ) -> None:
        """
        Use to create a VPU object.

        Parameters
        ----------
        device_id : str, optional
            The id of the device to use, by default None
            This can be a MXID, IP address, or USB port name.
            Examples: "14442C108144F1D000", "192.168.1.44", "3.3.3"

        """
        # general attributes
        self._mxid: str | None = device_id
        self._pipeline: dai.Pipeline = dai.Pipeline()
        self._thread: Thread | None = None
        self._start_condition = Condition()
        self._data_queue: Queue[list[np.ndarray | list[np.ndarray]]] = Queue()
        self._result_queue: Queue[list[dai.ADatatype | list[dai.ADatatype]]] = Queue()
        self._stopped = False

        # attributes for multi model execution
        self._blob_paths: list[str] = []
        self._inputnames: list[list[str] | None] = []
        self._nns: list[dai.node.NeuralNetwork] = []
        self._xins: list[dai.node.XLinkIn | list[dai.node.XLinkIn]] = []
        self._xouts: list[dai.node.XLinkOut] = []
        self._xin_names: list[str | list[str]] = []
        self._xout_names: list[str] = []

        # mode checking
        self._multimode: bool = False

        atexit.register(self.stop)

    def __del__(self: Self) -> None:
        """Use to stop the VPU."""
        self.stop()

    def __call__(
        self: Self,
        data: np.ndarray | list[np.ndarray] | list[np.ndarray | list[np.ndarray]],
        *,
        safe: bool | None = None,
    ) -> (
        dai.ADatatype | list[dai.ADatatype] | list[dai.ADatatype | list[dai.ADatatype]]
    ):
        """
        Use to run an inference on the VPU.

        Parameters
        ----------
        data : np.ndarray | list[np.ndarray]
            The data to run an inference on.
        safe : bool, optional
            If True, will evaluate the data before sending to the VPU.
            If False, will send data directly to the VPU.
            By default None, which will use safe mode.

        Returns
        -------
        dai.ADatatype | list[dai.ADatatype] | list[dai.ADatatype | list[dai.ADatatype]]
            The result of the inference.

        """
        return self.run(data, safe=safe)

    def stop(self: Self) -> None:
        """Use to stop the VPU."""
        self._stopped = True
        if self._thread is not None:
            if self._thread.is_alive():
                self._thread.join()
            else:
                pass

    def _reset(self: Self) -> None:
        """Reset the VPU attributes. Should only be called in reconfiguration."""
        # stop the thread if active
        if self._thread is not None:
            self.stop()
        self._stopped = False
        # reset list attributes
        self._blob_paths = []
        self._xins = []
        self._inputnames = []
        self._nns = []
        self._xouts = []
        self._xin_names = []
        self._xout_names = []
        # reset pipeline
        self._pipeline = dai.Pipeline()

    def reconfigure(
        self: Self,
        blob_path: str | Path,
        input_names: list[str] | None = None,
        model_data: YolomodelData | MobilenetData | None = None,
    ) -> None:
        """
        Use to reconfigure the VPU with a single new blob file.

        Parameters
        ----------
        blob_path : str | Path
            The path to the blob file.
        input_names : list[str], optional
            The names of the input layers. Defaults to None.
        model_data : YolomodelData | MobilenetData, optional
            The model data. Defaults to None.
            Can be used to set the YoloModelData or MobilenetData.
            If None, then a generic neural network will be created.

        """
        self._multimode = False
        # repackage as input to backend function
        self._reconfigure(
            [blob_path],
            [input_names],
            [model_data],
        )

    def reconfigure_multi(
        self: Self,
        blob_paths: list[str | Path],
        input_names: list[list[str] | None] | None = None,
        modeldata: list[YolomodelData | MobilenetData | None] | None = None,
    ) -> None:
        """
        Reconfigure the VPU with multiple blob files.

        Parameters
        ----------
        blob_paths : list[str | Path]
            The paths to the blob files.
        input_names : list[list[str] | None]
            The names of the input layers. Defaults to None.
            Should be filled in if a model has multiple inputs.
        modeldata : list[YolomodelData | MobilenetData | None]
            The model data. Defaults to None.
            Should be filled in if the model is a YOLO or Mobilenet model.
            If None, then a generic neural network will be created.

        """
        self._multimode = True
        self._reconfigure(
            blob_paths,
            input_names,
            modeldata,
        )

    def _reconfigure(
        self: Self,
        blob_paths: list[str | Path],
        input_names: list[list[str] | None] | None = None,
        modeldata: list[YolomodelData | MobilenetData | None] | None = None,
    ) -> None:
        """
        Handle reconfiguration of the VPU.

        Parameters
        ----------
        blob_paths : list[str | Path]
            The paths to the blob files.
        input_names : list[list[str] | None]
            The names of the input layers. Defaults to None.
            Should be filled in if a model has multiple inputs.
        modeldata : list[YolomodelData | MobilenetData | None]
            The model data. Defaults to None.

        Raises
        ------
        FileNotFoundError
            If a blob file does not exist.
        ValueError
            If input_names is not None and does not match the length of blob_paths.
        ValueError
            If modeldata is not None and does not match the length of blob_paths.
        ValueError
            If a yolo model does not have a single input.
        ValueError
            If a mobilenet model does not have a single input.
        TypeError
            If modeldata is not YoloModelData or MobilenetData.

        """
        # reset the VPU
        self._reset()

        self._blob_paths = [
            blob_path if isinstance(blob_path, str) else str(blob_path.resolve())
            for blob_path in blob_paths
        ]
        # validate the blob files
        for idx, blob_path in enumerate(self._blob_paths):
            if not Path.exists(Path(blob_path)):
                err_msg = f"Blob file #{idx}: {blob_path}, does not exist."
                raise FileNotFoundError(err_msg)

        # validate the input_names
        if input_names is not None:
            if len(input_names) != len(self._blob_paths):
                err_msg = "Input names must match the number of blob files."
                raise ValueError(err_msg)
        else:
            input_names = [None] * len(self._blob_paths)
        self._inputnames = input_names

        # validate the modeldata
        if modeldata is not None:
            if len(modeldata) != len(self._blob_paths):
                err_msg = "Model data must match the number of blob files."
                raise ValueError(err_msg)
        else:
            modeldata = [None] * len(self._blob_paths)

        # allocate each network in the pipeline
        for idx, (bpath, iname, mdata) in enumerate(
            zip(self._blob_paths, self._inputnames, modeldata),
        ):
            # allocate input XLink nodes
            if iname is None:
                xin_name = f"vpu_in_{idx}"
                self._xin_names.append(xin_name)
                self._xins.append(create_xin(self._pipeline, xin_name))
            else:
                m_xins = []
                m_xin_names = []
                for name in iname:
                    xin_name = f"vpu_in_{idx}_{name}"
                    m_xin_names.append(xin_name)
                    m_xins.append(create_xin(self._pipeline, name))
                self._xin_names.append(m_xin_names)
                self._xins.append(m_xins)

            # allocate neural network
            if mdata is None:
                # handle single or multi link
                links: list[dai.node.XLinkIn.Output] | dai.node.XLinkIn.Output = []
                if isinstance(self._xins[-1], list):
                    links = [xin.out for xin in self._xins[-1]]
                else:
                    links = self._xins[-1].out
                self._nns.append(
                    create_neural_network(
                        self._pipeline,
                        links,
                        Path(bpath),
                        iname,
                    ),
                )
            elif isinstance(mdata, YolomodelData):
                if isinstance(self._xins[-1], list):
                    err_msg = "Yolo model data must have a single input."
                    raise ValueError(err_msg)
                self._nns.append(
                    create_yolo_detection_network(
                        self._pipeline,
                        self._xins[-1].out,
                        Path(bpath),
                        yolo_data=mdata,
                    ),
                )
            elif isinstance(mdata, MobilenetData):
                if isinstance(self._xins[-1], list):
                    err_msg = "Mobilenet model data must have a single input."
                    raise ValueError(err_msg)
                self._nns.append(
                    create_mobilenet_detection_network(
                        self._pipeline,
                        self._xins[-1].out,
                        Path(bpath),
                        mobilenet_data=mdata,
                    ),
                )
            else:
                err_msg = "Model data must be YoloModelData or MobilenetData."
                raise TypeError(err_msg)

            # allocate output XLink nodes
            xout_name = f"vpu_out_{idx}"
            self._xout_names.append(xout_name)
            self._xouts.append(
                create_xout(self._pipeline, self._nns[-1].out, xout_name),
            )

        # create the thread
        self._thread = Thread(target=self._run_thread, daemon=True)
        self._thread.start()

        # wait for thread to start
        with self._start_condition:
            self._start_condition.wait()

    def _run_thread(
        self: Self,
    ) -> None:
        """
        Use in a thread to process neural networks on the VPU.

        Handles multi-input networks, and an arbitrary number of networks.
        Will pre-allocate all buffers and queues for each network.

        Raises
        ------
        RuntimeError
            If the pipeline is not set.
            If the data does not match queues and buffers.

        """
        # check pipeline, should always be set
        if self._pipeline is None:
            err_msg = "Pipeline not set."
            raise RuntimeError(err_msg)
        # create the device
        device_object = create_device(self._pipeline, device_id=self._mxid)
        with device_object as device:
            # pre-fetch queues and allocate buffers
            buffer: MultiBuffer = MultiBuffer(device, self._xin_names, self._xout_names)

            # notify the main thread that VPU is ready
            # this will allow the reconfigure call to return
            with self._start_condition:
                self._start_condition.notify()

            # loop until stopped
            while not self._stopped:
                if self._stopped:
                    break

                # get data
                try:
                    all_data: list[np.ndarray | list[np.ndarray]] = (
                        self._data_queue.get(timeout=0.1)
                    )
                except Empty:
                    continue

                # push data to networks
                buffer.send(all_data)

                # get the results
                self._result_queue.put(buffer.receive())

    def run(
        self: Self,
        data: np.ndarray | list[np.ndarray] | list[np.ndarray | list[np.ndarray]],
        *,
        safe: bool | None = None,
    ) -> (
        dai.ADatatype | list[dai.ADatatype] | list[dai.ADatatype | list[dai.ADatatype]]
    ):
        """
        Use to run an inference on the VPU.

        If the VPU was configured with multiple networks, then data must be a list of data.
        If the VPU was configured with a single network, then data can be a single np.ndarray
        or a list of np.ndarray if the network has multiple inputs.
        The return type will change based on whether the VPU was configured with multiple networks.
        If configured with a single network, then the return will be
        a single np.ndarray or dai.ImgDetections.
        If configured with multiple networks, then the return will be
        a list of np.ndarray or dai.ImgDetections.

        Parameters
        ----------
        data : np.ndarray | list[np.ndarray]
            The data to run an inference on.
        safe : bool, optional
            If True, will evaluate the data before sending to the VPU.
            If False, will send data directly to the VPU.
            By default None, which will use safe mode.

        Returns
        -------
        dai.ADatatype | list[dai.ADatatype] | list[dai.ADatatype | list[dai.ADatatype]]
            The result of the inference.

        Raises
        ------
        RuntimeError
            If the VPU thread is not set or alive. Will occur if not configured.
            If the VPU result is None.

        """
        if safe is None:
            safe = True
        if self._thread is None:
            err_msg = "VPU thread not set."
            raise RuntimeError(err_msg)
        if not self._thread.is_alive():
            err_msg = "VPU thread is not alive."
            raise RuntimeError(err_msg)
        _log.debug("VPU run called.")
        if not self._multimode:
            result = self._run_single(data, safe=safe)  # type: ignore[arg-type, assignment]
        else:
            result = self._run_multi(data, safe=safe)  # type: ignore[arg-type, assignment]
        return result

    def _run_single(
        self: Self,
        data: np.ndarray | list[np.ndarray],
        *,
        safe: bool | None = None,
    ) -> dai.ADatatype | list[dai.ADatatype]:
        if safe is None:
            safe = True
        if safe:
            # evaluate data when it is a list
            if isinstance(data, list):
                # in single mode must only contain np.array
                if any(isinstance(d, list) for d in data):
                    err_msg = "Configuration with single network expects np.array or list of np.arrays,"
                    err_msg += " not a list with lists inside."
                    raise TypeError(err_msg)
                if not all(isinstance(d, np.ndarray) for d in data):
                    err_msg = "Configuration with single network expects np.array or list of np.arrays."
                    raise TypeError(err_msg)
            self._data_queue.put([data])
        else:
            # just send it
            self._data_queue.put([data])
        _log.debug("Waiting on VPU result.")
        return self._result_queue.get()[0]

    def _run_multi(
        self: Self,
        data: list[np.ndarray | list[np.ndarray]],
        *,
        safe: bool | None = None,
    ) -> list[dai.ADatatype | list[dai.ADatatype]]:
        if safe is None:
            safe = True
        if safe:
            if isinstance(data, np.ndarray) or not isinstance(data, list):
                err_msg = "Configuration with multiple networks expects type: list[np.ndarray | list[np.ndarray]]."
                raise TypeError(err_msg)
            self._data_queue.put(data)
        else:
            self._data_queue.put(data)
        _log.debug("Waiting on VPU result.")
        return self._result_queue.get()
