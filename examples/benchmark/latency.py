# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Example showcasing getting the latency of the device."""

from __future__ import annotations

from oakutils.benchmark import measure_latency

mean_time, std_time, _ = measure_latency()
print(f"Mean latency: {mean_time:.3f} seconds")
print(f"Standard deviation: {std_time:.3f} seconds")
