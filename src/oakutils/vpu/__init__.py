"""
Submodule for using the onboard VPU as a standalone processor.

Classes
-------
VPU
    A class for using the onboard VPU as a standalone processor.
YolomodelData
    A dataclass for storing Yolomodel data.
MobilenetData
    A dataclass for storing Mobilenet data.
"""

from ._model_data import MobilenetData, YolomodelData
from ._vpu import VPU

__all__ = [
    "VPU",
    "YolomodelData",
    "MobilenetData",
]
