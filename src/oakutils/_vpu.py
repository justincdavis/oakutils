"""
Module for using the onboard VPU as a standalone processor.

Classes
-------
VPU
    A class for using the onboard VPU as a standalone processor.
"""
from __future__ import annotations

import atexit
import logging
import pathlib
from threading import Condition, Thread
from typing import TYPE_CHECKING

import depthai as dai

from .nodes import create_neural_network, create_xin, create_xout

if TYPE_CHECKING:
    import numpy as np
    from typing_extensions import Self

_log = logging.getLogger(__name__)


class VPU:
    """
    Class for using the onboard VPU as a standalone processor.

    Methods
    -------
    stop()
        Use to stop the VPU.
    reconfigure(blob_path, input_names=None)
        Use to reconfigure the VPU with a new blob file.
    run(data)
        Use to run an inference on the VPU.
    """

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

        atexit.register(self.stop)

    def __del__(self: Self) -> None:
        """Use to stop the VPU."""
        self.stop()

    def __call__(self: Self, data: np.ndarray | list[np.ndarray]) -> np.ndarray:
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
        return self.run(data)

    def stop(self: Self) -> None:
        """Use to stop the VPU."""
        self._stopped = True
        with self._condition:
            self._condition.notify()
        if self._thread is not None:
            if self._thread.is_alive():
                self._thread.join()
            else:
                pass

    def reconfigure(
        self: Self,
        blob_path: str,
        input_names: list[str] | None = None,
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
        # stop the VPU if it is running
        if self._thread is not None:
            self.stop()
        self._stopped = False
        self._blob_path = blob_path
        # create pipeline with neural network
        self._pipeline = dai.Pipeline()
        if input_names is None:
            _log.debug("Reconfiguring VPU with single input.")
            self._xin = create_xin(self._pipeline, "vpu_in")
            self._nn = create_neural_network(
                self._pipeline,
                self._xin.out,
                pathlib.Path(self._blob_path),
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
                pathlib.Path(self._blob_path),
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
        if self._pipeline is None:
            raise RuntimeError("Pipeline not set.")
        with dai.Device(self._pipeline) as device:
            _log.debug("VPU thread started.")
            with self._start_condition:
                self._start_condition.notify()
            while not self._stopped:
                with self._condition:
                    self._condition.wait()
                _log.debug("VPU thread notified.")
                if self._stopped:
                    _log.debug("VPU stopped, breaking.")
                    break
                if isinstance(self._xin, list):
                    _log.debug("Sending multi-value data to VPU.")
                    if self._input_names is None:
                        raise RuntimeError("Input names not set.")
                    if self._data is None or not isinstance(self._data, list):
                        raise RuntimeError("Data not set or data is not a list.")
                    for name, data in zip(self._input_names, self._data):
                        buff = dai.Buffer()
                        buff.setData(data)
                        device.getInputQueue(name).send(buff)  # type: ignore[attr-defined]
                else:
                    _log.debug("Sending single-value data to VPU.")
                    buff = dai.Buffer()
                    # setData takes list[int] or ndarray, mypy cannot handle this
                    buff.setData(self._data)  # type: ignore[arg-type]
                    device.getInputQueue("vpu_in").send(buff)  # type: ignore[attr-defined]
                self._result = device.getOutputQueue("vpu_out").get()  # type: ignore[attr-defined]
                _log.debug("VPU result received, notifying primary thread.")
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
            If the VPU thread is not set or alive. (Should not happen.)
            If the VPU result is None.
        """
        if self._blob_path is None:
            raise RuntimeError("Blob path not set.")
        if self._thread is None:
            raise RuntimeError("VPU thread not set.")
        if not self._thread.is_alive():
            raise RuntimeError("VPU thread is not alive.")
        _log.debug("VPU run called.")
        self._data = data
        with self._condition:
            self._condition.notify()
        _log.debug("Notified VPU thread, waiting for result.")
        with self._condition:
            self._condition.wait()
        if self._result is None:
            raise RuntimeError("VPU result is None.")
        return self._result
