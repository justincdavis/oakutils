# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
Models for the laserscan node.

Functions
---------
create_laserscan
    Creates a laserscan model as a node.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ._load import create_width_model as _create_width_model

if TYPE_CHECKING:
    import depthai as dai


def create_laserscan(
    pipeline: dai.Pipeline,
    input_link: dai.Node.Output,
    width: int = 10,
    shaves: int = 1,
) -> dai.node.NeuralNetwork:
    """
    Use to create a laserscan model with a specified width.

    Parameters
    ----------
    pipeline : dai.Pipeline
        The pipeline to add the laserscan to
    input_link : dai.node.XLinkOut
        The input link to connect to the laserscan node.
        Example: stereo.depth.link
        Explicitly pass in the link as a non-called function.
    width : int, optional
        The width of the laserscan, by default 10
        Options are [5, 10, 20]
    shaves : int, optional
        The number of shaves to use, by default 1
        Must be between 1 and 6

    Returns
    -------
    dai.node.NeuralNetwork
        The laserscan node

    Raises
    ------
    ValueError
        If the width is invalid

    """
    model_type = "laserscan"
    valid_widths: list[int] = [5, 10, 20]
    if width not in valid_widths:
        err_msg = "Invalid width, must be one of [5, 10, 20]"
        raise ValueError(err_msg)

    return _create_width_model(
        pipeline=pipeline,
        input_link=input_link,
        model_name=model_type,
        width=width,
        shaves=shaves,
    )
