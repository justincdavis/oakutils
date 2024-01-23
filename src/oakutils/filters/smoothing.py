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
Module for filters which smooth numpy arrays.

Classes
-------
ExpSmooth
    A class for exponentially smoothing numpy arrays.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from typing_extensions import Self


class ExpSmooth:
    """
    A class for exponentially smoothing numpy arrays.

    Attributes
    ----------
    alpha : float
        The alpha parameter for the exponential smoothing.


    Methods
    -------
    smooth
        Use to smooth a numpy array.
    """

    def __init__(self: Self, alpha: float = 0.9) -> None:
        """
        Use to create an ExpSmooth object.

        Parameters
        ----------
        alpha : float
            The alpha parameter for the exponential smoothing.
            Defaults to 0.9.
        """
        self._alpha = alpha

    @property
    def alpha(self: Self) -> float:
        """
        The alpha parameter for the exponential smoothing.

        Returns
        -------
        float
            The alpha parameter for the exponential smoothing.
        """
        return self._alpha

    @alpha.setter
    def alpha(self: Self, alpha: float) -> None:
        """
        Use to set the alpha parameter for the exponential smoothing.

        Parameters
        ----------
        alpha : float
            The alpha parameter for the exponential smoothing.
        """
        self._alpha = alpha

    def smooth(self: Self, data: list[np.ndarray]) -> np.ndarray:
        """
        Use to smooth a numpy array.

        Parameters
        ----------
        data : list[np.ndarray]
            The data to smooth.


        Returns
        -------
        np.ndarray
            The smoothed data.
        """
        smoothed = data[0]
        for frame in data[1:]:
            smoothed = smoothed * self.alpha + frame * (1 - self.alpha)
        return smoothed
