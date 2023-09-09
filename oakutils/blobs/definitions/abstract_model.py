from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from .utils.types import InputType, ModelType


class AbstractModel(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()

    @classmethod
    @abstractmethod
    def model_type(cls) -> ModelType:
        """The type of input this model takes."""
        pass

    @classmethod
    @abstractmethod
    def input_names(cls) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        pass

    @classmethod
    @abstractmethod
    def output_names(cls) -> list[str]:
        """The names of the output tensors."""
        pass
