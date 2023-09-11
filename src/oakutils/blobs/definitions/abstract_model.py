from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from typing_extensions import Self

    from .utils.types import InputType, ModelType


class AbstractModel(ABC, torch.nn.Module):
    def __init__(self: Self) -> None:
        super().__init__()

    @classmethod
    @abstractmethod
    def model_type(cls: AbstractModel) -> ModelType:
        """The type of input this model takes."""

    @classmethod
    @abstractmethod
    def input_names(cls: AbstractModel) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""

    @classmethod
    @abstractmethod
    def output_names(cls: AbstractModel) -> list[str]:
        """The names of the output tensors."""
