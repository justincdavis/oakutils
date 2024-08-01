# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Create script nodes on the OAK device.

Functions
---------
create_script
    Use to create a script node
get_available_imports
    Get the available imports for the script node
verify_script
    Verify the imports of a script

"""

from __future__ import annotations

import re
import time
from pathlib import Path

import depthai as dai

from ._script_utils import AVAILABLE_LIBRARIES as _AVAILABLE_LIBRARIES
from ._script_utils import AVAILABLE_MODULES as _AVAILABLE_MODULES


def get_available_imports() -> list[str]:
    """
    Get the available imports for the script node.

    Returns
    -------
    list[str]
        The available imports for the script node.

    """
    return _AVAILABLE_LIBRARIES + _AVAILABLE_MODULES


def verify_script(
    script: str | Path,
) -> tuple[bool, str]:
    """
    Verify the imports of a script.

    Parameters
    ----------
    script : str | Path
        The script to verify.
        If the type is a string then it is used as the script.
        If a pathlib.Path is given, then the Path should
        be a file containing the script.

    Returns
    -------
    tuple[bool, str]
        Success, message (message is empty if success is true)

    Raises
    ------
    FileNotFoundError
        If the script file does not exist.

    """
    script_txt = ""
    if isinstance(script, Path):
        if not script.exists():
            err_msg = f"Script file {script} does not exist."
            raise FileNotFoundError(err_msg)
        script_txt = script.read_text()
    else:
        script_txt = script

    # PART 1
    # evaluate if there are syntax errors
    try:
        compile(script_txt, "<string>", "exec")
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except ValueError as e:
        return False, f"Value error: {e}"

    # PART 2
    # now check if any
    # import *
    # OR
    # from * import *
    # statements exists, and verify the * is a valid library/module
    matches: list[list[tuple[str, str]]] = [
        re.findall(
            r"(?m)^(?:from[ ]+(\S+)[ ]+)?import[ ]+(\S+)(?:[ ]+as[ ]+\S+)?[ ]*$",
            line,
        )
        for line in script_txt.split("\n")
    ]
    # prune empty matches
    num_matches = len(matches)
    for i in range(num_matches - 1, -1, -1):
        if not matches[i]:
            matches.pop(i)
    # unnest the matches
    unnested_matches: list[tuple[str, str]] = []
    for m in matches:
        unnested_matches.extend(m)
    # compare againist the available imports
    imports = get_available_imports()
    for match in unnested_matches:
        lib = match[0]
        if not lib:
            lib = match[1]
        if lib not in imports:
            return False, f"Invalid import {lib}."
    return True, ""


def create_script(
    pipeline: dai.Pipeline,
    script: str | Path,
    name: str | None = None,
    processor: dai.ProcessorType = dai.ProcessorType.LEON_CSS,
    *,
    verify: bool | None = None,
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
    verify : bool, optional
        Verify the script imports before creating the node, by default False

    Returns
    -------
    dai.node.Script
        The script node

    Raises
    ------
    ValueError
        If there are issues with the script syntax or imports
    FileNotFoundError
        If the script file does not exist

    """
    if verify:
        success, err_msg = verify_script(script)
        if not success:
            raise ValueError(err_msg)

    name = name or f"script_{time.monotonic_ns()}"
    script_node = pipeline.create(dai.node.Script)
    script_node.setProcessor(processor)
    if isinstance(script, Path):
        if not script.exists():
            err_msg = f"Script file {script} does not exist."
            raise FileNotFoundError(err_msg)
        script_node.setScriptPath(path=script, name=name)
    else:
        script_node.setScript(script=script, name=name)
    return script_node
