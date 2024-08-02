# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Example showcasing using the BlobEvaluator."""

from __future__ import annotations

from oakutils import set_log_level
from oakutils.blobs.models.bulk import GAUSSIAN_3X3
# from oakutils.blobs.models.shave1 import GAUSSIAN_3X3 as gauss1
# from oakutils.blobs.models.shave2 import GAUSSIAN_3X3 as gauss2
# from oakutils.blobs.models.shave3 import GAUSSIAN_3X3 as gauss3
# from oakutils.blobs.models.shave4 import GAUSSIAN_3X3 as gauss4
# from oakutils.blobs.models.shave5 import GAUSSIAN_3X3 as gauss5
# from oakutils.blobs.models.shave6 import GAUSSIAN_3X3 as gauss6
from oakutils.blobs.testing import BlobEvaluater


set_log_level("ERROR")

blob_eval = BlobEvaluater([*GAUSSIAN_3X3])

results = blob_eval.run()
