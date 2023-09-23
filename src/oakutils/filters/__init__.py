"""
Module for performing filtering on typical datatypes.

Submodules
----------
wls
    Module for performing WLS filtering.

Classes
-------
WLSFilter
    A class for computing the weighted-least-squares filter,
    on disparity images.
"""
from . import wls
from .wls import WLSFilter

__all__ = [
    "wls",
    "WLSFilter",
]
