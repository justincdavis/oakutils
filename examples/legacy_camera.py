# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Example showcasing how to use the LegacyCamera abstraction."""

from __future__ import annotations

import time

from oakutils import LegacyCamera

cam = LegacyCamera(
    display_depth=True,
)
cam.start(block=True)

time.sleep(10)

cam.stop()
