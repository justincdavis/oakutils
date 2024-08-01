# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Premade scripts for allowing more advanced usage.

Classes
-------
Router
    Class implementing a switching router for a one-to-many mapping.

References
----------
https://docs.luxonis.com/software/depthai/examples/script_change_pipeline_flow/


"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import depthai as dai

from .script import create_script
from .xin import create_xin
from .xout import create_xout

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self


class Router:
    """
    Class wrapper for a script which routes a message between nodes.

    The router acts as a one-to-many mapping where a single input message
    can be 'routed' to a single output link. The output link is determined
    by setting the index or key (string name) of the output link.
    """

    def __init__(
        self: Self,
        pipeline: dai.Pipeline,
        input_link: dai.Node.Output,
        links: Sequence[dai.Node.Input | tuple[dai.Node.Input, str]],
        start_index: str | int = 0,
        router_name: str | None = None,
        processor: dai.ProcessorType = dai.ProcessorType.LEON_CSS,
    ) -> None:
        """
        Create a router script.

        Parameters
        ----------
        pipeline : dai.Pipeline
            The pipeline to add the router to.
        input_link : dai.Node.Output
            The input link to route messages from.
        links : Sequence[dai.Node.Input | tuple[dai.Node.Input, str]]
            The links to route to.
            If a tuple is given, the first element is the link and the second
            element is the name of the link. The name is used to re-route
            message via strings instead of indices.
        start_index : str | int
            The index or key to start the router at.
            Default is 0.
        router_name : str | None
            The name of the router.
            If None, the name will be 'router_{timestamp}'.
        processor : dai.ProcessorType
            The processor to run the script on.
            Default is dai.ProcessorType.LEON_CSS.

        """
        self._name = router_name or f"router_{time.monotonic_ns()}"
        self._msg_key = f"{self._name}_xout_msg"
        self._xout = create_xout(pipeline, input_link, self._msg_key)
        self._index_key = f"{self._name}_xin_index"
        self._index_xin = create_xin(pipeline, self._index_key)
        self._toggle_key = f"{self._name}_xin_toggle"
        self._toggle_xin = create_xin(pipeline, self._toggle_key)
        self._toggle = True
        self._keys: dict[str, int] = {}

        self._pure_links: list[tuple[dai.Node.Input, str]] = []
        for idx, link in enumerate(links):
            linkname = f"link_{idx}"
            if isinstance(link, tuple):
                _link, _key = link
                self._keys[_key] = idx
                self._pure_links.append((_link, linkname))
            else:
                self._pure_links.append((link, linkname))
        self._max_index = len(self._pure_links)

        if isinstance(start_index, str):
            self._start_index = self._keys.get(start_index, 0)
        else:
            self._start_index = start_index

        base_script = f"""
            index = {self._start_index}
            toggle = True
            while True:
                toggle_msg = node.io[{self._toggle_key}].get()
                if toggle_msg is not None:
                    toggle = toggle_msg.getData()[0]

                idx_msg = node.io[{self._index_key}].tryGet()
                if idx_msg is not None:
                    index = idx_msg.getData()[0]

                data_msg = node.io[{self._msg_key}].get()

                if toggle:
        """
        for idx, (_, lname) in enumerate(self._pure_links):
            base_script += f"""
                    if index == {idx}:
                        node.io[{lname}].send(data_msg)
            """

        # Finally create the script and link the xin
        self._script = create_script(
            pipeline=pipeline,
            script=base_script,
            name=self._name,
            processor=processor,
            verify=True,
        )
        self._index_xin.out.link(self._script.inputs[self._index_key])
        self._toggle_xin.out.link(self._script.inputs[self._toggle_key])
        for plink, lname in self._pure_links:
            self._script.outputs[lname].link(plink)

        # runtime variables
        self._index_input_queue: dai.DataInputQueue | None = None
        self._toggle_input_queue: dai.DataInputQueue | None = None

    def setup(self: Self, device: dai.DeviceBase) -> None:
        """
        Set the router up once the device has been started.

        Parameters
        ----------
        device : dai.DeviceBase
            The device to setup the router on.

        """
        self._index_input_queue = device.getInputQueue(self._index_key)  # type: ignore[attr-defined]
        self._toggle_input_queue = device.getInputQueue(self._toggle_key)  # type: ignore[attr-defined]

    def toggle(self: Self, *, on: bool | None = None) -> None:
        """
        Toggle the flow of data through the router.

        Parameters
        ----------
        on : bool | None
            If True, then the router will allow data to flow.
            If False, then the router will stop data from flowing.
            If None, then the router will toggle the flow.

        Raises
        ------
        RuntimeError
            If the router has not been setup.

        """
        if self._toggle_input_queue is None:
            err_msg = "Router has not been setup."
            raise RuntimeError(err_msg)

        if on is not None:
            self._toggle = on
        else:
            self._toggle = not self._toggle
        buffer = dai.Buffer()
        buffer.setData([int(self._toggle)])
        self._toggle_input_queue.send(buffer)

    def route(self: Self, key: str | int) -> None:
        """
        Route the message to the specified link.

        Parameters
        ----------
        key : str | int
            The key or index of the link to route the message to.

        Raises
        ------
        ValueError
            If the key is invalid.
        RuntimeError
            If the router has not been setup.

        """
        if self._index_input_queue is None:
            err_msg = "Router has not been setup."
            raise RuntimeError(err_msg)

        if isinstance(key, str):
            index = self._keys.get(key, None)
            if index is None:
                err_msg = f"Invalid key {key}."
                raise ValueError(err_msg)
        else:
            index = key

        if index >= self._max_index:
            err_msg = f"Invalid index {index}."
            raise ValueError(err_msg)

        buffer = dai.Buffer()
        buffer.setData([index])
        self._index_input_queue.send(buffer)
