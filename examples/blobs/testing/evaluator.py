# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Example showcasing using the BlobEvaluator."""

from __future__ import annotations

from pathlib import Path

import cv2
from oakutils.blobs.testing import BlobEvaluater
from oakutils.blobs import compile_model
from oakutils.blobs.definitions import Gaussian


blob_paths = [
    compile_model(Gaussian, {}, s) for s in range(1, 7)
]
blob_eval = BlobEvaluater(blob_paths)

image = cv2.imread(str(Path("data/test.png").resolve()))

for data in [image, None]:
    results = blob_eval.run(data)
    for result in results:
        print(result)

    print(blob_eval.allclose())
