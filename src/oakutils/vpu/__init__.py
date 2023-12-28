"""
Submodule for using the onboard VPU as a standalone processor.
"""

from ._model_data import YolomodelData, MobilenetData
from ._vpu import VPU

__all__ = [
    "VPU",
    "YolomodelData",
    "MobilenetData",
]
