# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
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
    "MobilenetData",
    "YolomodelData",
]
