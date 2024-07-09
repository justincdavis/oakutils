# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Example showcasing getting the throughput of the device."""

from __future__ import annotations

from oakutils.benchmark import measure_throughput

downlink, uplink = measure_throughput()
print(f"Downlink throughput: {downlink:.3f} MB/s")
print(f"Uplink throughput: {uplink:.3f} MB/s")
