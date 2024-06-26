# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
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
