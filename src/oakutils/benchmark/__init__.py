# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Submodule for benchmarking characteristics of OAK devices.

Functions
---------
measure_latency
    Measure the latency of communication with the device.
measure_throughput
    Measure the throughput of the devices connection.

"""

from __future__ import annotations

from ._latency import measure_latency
from ._throughput import measure_throughput

__all__ = ["measure_latency", "measure_throughput"]
