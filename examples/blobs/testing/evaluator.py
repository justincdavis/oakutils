# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Example showcasing using the BlobEvaluator."""

from __future__ import annotations

import numpy as np
from oakutils.blobs.models.bulk import GAUSSIAN_3X3
from oakutils.blobs.testing import BlobEvaluater
from oakutils.nodes import get_nn_bgr_frame


blob_eval = BlobEvaluater([*GAUSSIAN_3X3])

results = blob_eval.run()
for result in results:
    print(result)

results = [
    get_nn_bgr_frame(r) for r in results
]

for idx in range(len(results)):
    if not np.allclose(results[0], results[idx]):
        raise ValueError("Results do not match.")
else:
    print("Results match.")
