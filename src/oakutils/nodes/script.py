# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Create script nodes on the OAK device.

Functions
---------
create_script
    Use to create a script node

"""

from __future__ import annotations

import time
from pathlib import Path

import depthai as dai


def verify_script(
    script: str | Path,
) -> bool:
    """
    """

def create_script(
    pipeline: dai.Pipeline,
    script: str | Path,
    name: str | None = None,
    processor: dai.ProcessorType = dai.ProcessorType.LEON_CSS,
) -> dai.node.Script:
    """
    Use to create a script node.

    Parameters
    ----------
    pipeline : dai.Pipeline
        The pipeline to add the script node to
    script : str | Path
        The script to run on the device.
        If the type is a string, then the script will be
        used directly to create the node.
        If a pathlib.Path is given, then the Path should
        be a file containing the script.
    name : str | None, optional
        The name of the script node, by default None
        If None, then the name will be timestamped.
        script_{timestamp}
    processor : dai.ProcessorType, optional
        The processor type to run the script on, by default dai.ProcessorType.LEON_CSS
        Should only be changed if you know what you are doing.

    Returns
    -------
    dai.node.Script
        The script node

    """
    name = name if name else f"script_{time.time()}"
    script_node = pipeline.create(dai.node.Script)
    script_node.setProcessor(processor)
    if isinstance(script, Path):
        script_node.setScriptPath(path=script, name=name)
    else:
        script_node.setScript(script=script, name=name)
    return script_node
