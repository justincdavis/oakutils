# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Example showcasing benchmarking of a YOLO model on VPU."""

from __future__ import annotations

from pathlib import Path

from oakutils.blobs import benchmark_blob


def main() -> None:
    yolo_path = Path(__file__).parent / "yolov8n_160"
    result = benchmark_blob(yolo_path, is_yolo=True)
    print(result)


if __name__ == "__main__":
    main()
