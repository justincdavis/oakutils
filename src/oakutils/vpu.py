"""
Module for using the onboard VPU as a standalone processor.

Classes
-------
VPU
    A class for using the onboard VPU as a standalone processor.
"""
from __future__ import annotations

import logging
from threading import Condition, Thread
from typing import TYPE_CHECKING

import depthai as dai

from .nodes import create_neural_network, create_xin, create_xout

if TYPE_CHECKING:
    import numpy as np
    from typing_extensions import Self

_log = logging.getLogger(__name__)


class VPU:
    """Class for using the onboard VPU as a standalone processor."""

    def __init__(self: Self) -> None:
        """
        Use to create a VPU object.

        Parameters
        ----------
        blob_path : str
            The path to the blob file.
        """
        self._blob_path: str | None = None
        self._pipeline: dai.Pipeline | None = None
        self._xin: dai.node.XLinkIn | list[dai.node.XLinkIn] | None = None
        self._input_names: list[str] | None = None
        self._nn: dai.node.NeuralNetwork | None = None
        self._xout: dai.node.XLinkOut | None = None
        self._thread: Thread | None = None
        self._start_condition = Condition()
        self._condition = Condition()
        self._stopped = False
        self._data: np.ndarray | list[np.ndarray] | None = None
        self._result: np.ndarray | None = None

    def stop(self: Self) -> None:
        """Use to stop the VPU."""
        self._stopped = True
        with self._condition:
            self._condition.notify()
        self._thread.join()

    def reconfigure(
        self: Self, blob_path: str, input_names: list[str] | None = None, input_size: int = 1
    ) -> None:
        """
        Use to reconfigure the VPU with a new blob file.

        Parameters
        ----------
        blob_path : str
            The path to the blob file.
        input_names : list[str]
            The names of the input layers. Defaults to None.
        """
        self._blob_path = blob_path
        # create pipeline with neural network
        self._pipeline = dai.Pipeline()
        if input_names is None:
            _log.debug("Reconfiguring VPU with single input.")
            self._xin = create_xin(self._pipeline, "vpu_in")
            self._nn = create_neural_network(
                self._pipeline,
                self._xin.out,
                self._blob_path,
            )
            self._xout = create_xout(self._pipeline, self._nn.out, "vpu_out")
        else:
            _log.debug("Reconfiguring VPU with multiple inputs.")
            self._input_names = input_names
            self._xin = []
            for name in self._input_names:
                self._xin.append(create_xin(self._pipeline, name))
            self._nn = create_neural_network(
                self._pipeline,
                [xin.out for xin in self._xin],
                self._blob_path,
                self._input_names,
            )
            self._xout = create_xout(self._pipeline, self._nn.out, "vpu_out")
        # reallocate the device thread
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()
        with self._start_condition:
            self._start_condition.wait()

    def _run(self: Self) -> None:
        """Use in a thread to process the data on the VPU."""
        with dai.Device(self._pipeline) as device:
            _log.debug("VPU thread started.")
            with self._start_condition:
                self._start_condition.notify()
            while not self._stopped:
                with self._condition:
                    self._condition.wait()
                if self._stopped:
                    _log.debug("VPU stopped, breaking.")
                    break
                if isinstance(self._xin, list):
                    _log.debug("Sending multi-value data to VPU.")
                    for name, data in zip(self._input_names, self._data):
                        buff = dai.Buffer()
                        buff.setData(data)
                        device.getInputQueue(name).send(buff)
                else:
                    _log.debug("Sending single-value data to VPU.")
                    buff = dai.Buffer()
                    buff.setData(self._data)
                    device.getInputQueue("vpu_in").send(buff)
                self._result = device.getOutputQueue("vpu_out").get()
                with self._condition:
                    self._condition.notify()

    def run(self: Self, data: np.ndarray | list[np.ndarray]) -> np.ndarray:
        """
        Use to run an inference on the VPU.

        Parameters
        ----------
        data : np.ndarray | list[np.ndarray]
            The data to run an inference on.


        Returns
        -------
        np.ndarray
            The result of the inference.

        Raises
        ------
        RuntimeError
            If the blob path is not set.
        """
        if self._blob_path is None:
            raise RuntimeError("Blob path not set.")
        if not self._thread.is_alive():
            raise RuntimeError("VPU thread is not alive.")
        _log.debug("VPU run called.")
        self._data = data
        with self._condition:
            self._condition.notify()
        _log.debug("Notified VPU thread, waiting for result.")
        with self._condition:
            self._condition.wait()
        return self._result
