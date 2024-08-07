# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Example showcasing using the BlobEvaluator."""

from __future__ import annotations

from oakutils import set_log_level
from oakutils.blobs.testing import BlobEvaluater
from oakutils.blobs.models.bulk import LASERSCAN_20_1, POINTCLOUD


set_log_level("WARNING")
for blobset in [LASERSCAN_20_1, POINTCLOUD]:
    blob_paths = [*blobset]
    blob_eval = BlobEvaluater(blob_paths)

    results = blob_eval.run()
    for result in results:
        print(result)

    print(blob_eval.allclose(image_output=False, u8_input=True))
