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
Module for the AbstractModel class.

Classes
-------
AbstractModel
    Abstract base class for models.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from typing_extensions import Self

    from .utils.types import InputType, ModelType


class AbstractModel(ABC, torch.nn.Module):
    """Class defining the interface for models."""

    def __init__(self: Self) -> None:
        """Use to create an instance of the model."""
        super().__init__()

    @classmethod
    @abstractmethod
    def model_type(cls: type[AbstractModel]) -> ModelType:
        """Use to get the type of input this model takes."""

    @classmethod
    @abstractmethod
    def input_names(cls: type[AbstractModel]) -> list[tuple[str, InputType]]:
        """Use to get the names of the input tensors."""

    @classmethod
    @abstractmethod
    def output_names(cls: type[AbstractModel]) -> list[str]:
        """Use to get the names of the output tensors."""
