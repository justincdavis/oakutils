# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Example showcasing using the BlobEvaluator."""

from __future__ import annotations

from pathlib import Path

import cv2
from oakutils.blobs.testing import BlobEvaluater
from oakutils.blobs.models.bulk import GAUSSIAN_15X15


blob_paths = [*GAUSSIAN_15X15]
blob_eval = BlobEvaluater(blob_paths)

image = cv2.imread(str(Path("data/test.png").resolve()))

for data in [image, None]:
    results = blob_eval.run(data)
    for result in results:
        print(result)

    print(blob_eval.allclose())
