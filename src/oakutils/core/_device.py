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
from __future__ import annotations

import depthai as dai


def create_device(
    pipeline: dai.Pipeline,
    device_id: str | None = None,
) -> dai.DeviceBase:
    """
    Create a DepthAI device object from a pipeline.

    Parameters
    ----------
    pipeline : dai.Pipeline
        The pipeline to use
    device_id : str, optional
        The id of the device to use, by default None
        This can be a MXID, IP address, or USB port name.
        Examples: "14442C108144F1D000", "192.168.1.44", "3.3.3"

    Returns
    -------
    dai.Device
        The DepthAI device object

    """
    if device_id is not None:
        device_info: dai.DeviceInfo = dai.DeviceInfo(device_id)
        device_object = dai.Device(pipeline, device_info)
    else:
        device_object = dai.Device(pipeline)

    return device_object
