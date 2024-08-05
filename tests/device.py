# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from functools import lru_cache

import depthai as dai


@lru_cache
def get_device_count() -> int:
    """Get the number of connected devices."""
    return len(dai.Device.getAllAvailableDevices())
